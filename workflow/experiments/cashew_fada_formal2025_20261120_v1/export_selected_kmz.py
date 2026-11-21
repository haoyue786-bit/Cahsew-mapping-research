"""Render selected four-class rasters as full-resolution KML superoverlays."""

from collections import Counter
import io
import posixpath
from xml.etree import ElementTree
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
from PIL import Image, ImageDraw
from rasterio.coords import BoundingBox

from tif_class2_to_kmz import LEVELS, TILE_HEIGHT, TILE_WIDTH, region_xml, tile_kml
from export_selected import NODATA, PALETTE, UNKNOWN


NS = 'http://www.opengis.net/kml/2.2'
TEXT = {0: 'Cropland and other open land', 1: 'Other trees, wetlands and water',
        2: 'Cashew', 3: 'Built-up', UNKNOWN: 'Uncertain (Cashew below 0.5)'}


def png_bytes(image, **options):
    with io.BytesIO() as buffer:
        image.save(buffer, format='PNG', compress_level=3, **options)
        return buffer.getvalue()


def legend(group, uncertain):
    labels = tuple(TEXT) if uncertain else (0, 1, 2, 3)
    image = Image.new('RGBA', (258, 24 + 18 * len(labels)), (255, 255, 255, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline='#30353b')
    draw.text((8, 5), group.replace('_', ' '), fill='#20242a')
    for i, code in enumerate(labels):
        y = 21 + i * 18
        draw.rectangle((8, y, 21, y + 12), fill=PALETTE[code], outline='#43484f')
        draw.text((28, y), f'{code}  {TEXT[code]}', fill='#20242a')
    return image


def document(report, group, uncertain, size):
    tiles = report['tiles']
    west = min(tile['bounds'][0] for tile in tiles.values())
    south = min(tile['bounds'][1] for tile in tiles.values())
    east = max(tile['bounds'][2] for tile in tiles.values())
    north = max(tile['bounds'][3] for tile in tiles.values())
    name = f"{group} · selected epoch {report['epoch']} · four-class argmax"
    description = 'Each pixel uses the class with highest probability.'
    if uncertain:
        name += ' · Cashew 0.5'
        description = 'Argmax Cashew with probability below 0.5 is class 10 (uncertain).'
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<kml xmlns="{NS}"><Document><name>{escape(name)}</name><open>1</open>',
             f'<description>{escape(description)} Pixels without imagery are transparent.</description>',
             f'<LookAt><longitude>{(west+east)/2:.9f}</longitude><latitude>{(south+north)/2:.9f}</latitude>'
             f'<range>{max(east-west,north-south)*150000:.1f}</range></LookAt>',
             '<ScreenOverlay><name>Legend</name><Icon><href>legend.png</href></Icon>'
             '<overlayXY x="0" y="0" xunits="fraction" yunits="fraction"/>'
             '<screenXY x="0.015" y="0.035" xunits="fraction" yunits="fraction"/>'
             f'<size x="{size[0]}" y="{size[1]}" xunits="pixels" yunits="pixels"/></ScreenOverlay>']
    for tile, row in tiles.items():
        parts.append(f'<NetworkLink><name>Tile {tile}</name>{region_xml(*row["bounds"], 0, -1)}'
                     f'<Link><href>overlays/{tile}/0/0/0.kml</href><viewRefreshMode>onRegion</viewRefreshMode>'
                     '</Link></NetworkLink>')
    return ''.join(parts + ['</Document></kml>']).encode()


def render(workspace, report, group, uncertain, path):
    tiles = report['tiles']
    rank = {tile: i for i, tile in enumerate(sorted(tiles, key=lambda t: (tiles[t]['patches'], t)))}
    filename_suffix = 'p05' if uncertain else 'argmax'
    palette = [0] * 768
    for code, color in PALETTE.items():
        palette[3*code:3*code+3] = color
    full_grid = 2 ** (LEVELS - 1)
    card = legend(group, uncertain)
    with ZipFile(path, 'x', compression=ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        archive.writestr('doc.kml', document(report, group, uncertain, card.size))
        archive.writestr('legend.png', png_bytes(card), compress_type=ZIP_STORED)
        for tile, row in tiles.items():
            codes = np.load(workspace / f'{tile}_{filename_suffix}.npy', mmap_mode='r', allow_pickle=False)
            if codes.dtype != np.uint8 or codes.shape != (full_grid * TILE_HEIGHT, full_grid * TILE_WIDTH):
                raise ValueError('Unexpected full-resolution KMZ input array')
            for level in range(LEVELS):
                grid = 2 ** level
                stride = full_grid // grid
                preview = codes[stride//2::stride, stride//2::stride]
                for x in range(grid):
                    left = x * TILE_WIDTH
                    for y in range(grid):
                        top = (grid - 1 - y) * TILE_HEIGHT
                        image = Image.fromarray(preview[top:top+TILE_HEIGHT, left:left+TILE_WIDTH], mode='P')
                        image.putpalette(palette)
                        name = f'overlays/{tile}/{level}/{x}/{y}'
                        xml = tile_kml(f'{group} · {tile[-4:]}', level, x, y, BoundingBox(*row['bounds']))
                        xml = xml.replace(f'<drawOrder>{level}</drawOrder>'.encode(),
                                          f'<drawOrder>{level * len(tiles) + rank[tile]}</drawOrder>'.encode())
                        archive.writestr(name + '.kml', xml)
                        archive.writestr(name + '.png', png_bytes(image, transparency=NODATA), compress_type=ZIP_STORED)


def verify_archive(path, report, uncertain):
    tiles = report['tiles']
    expected = 2 * (1 + len(tiles) * sum(4 ** level for level in range(LEVELS)))
    expected_by_tile = {tile: row['uncertain_pixels'] if uncertain else 0 for tile, row in tiles.items()}
    uncertainty = Counter()
    valid_codes = {0, 1, 2, 3, NODATA} | ({UNKNOWN} if uncertain else set())
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        if (archive.testzip() or len(names) != len(archive.namelist()) or len(names) != expected
                or not {'doc.kml', 'legend.png'} <= names):
            raise ValueError('KML archive is incomplete or invalid')
        for name in names:
            if name.endswith('.kml'):
                tree = ElementTree.fromstring(archive.read(name))
                for href in tree.iter(f'{{{NS}}}href'):
                    target = posixpath.normpath(posixpath.join(posixpath.dirname(name), href.text))
                    if target not in names:
                        raise ValueError(f'Dangling KML link {name}: {target}')
            elif name.startswith('overlays/') and int(name.split('/')[2]) == LEVELS - 1:
                with Image.open(io.BytesIO(archive.read(name))) as image:
                    if (image.mode != 'P' or image.info.get('transparency') != NODATA
                            or image.size != (TILE_WIDTH, TILE_HEIGHT)):
                        raise ValueError('Invalid classification overlay')
                    values = image.histogram()
                    if not {i for i, n in enumerate(values) if n} <= valid_codes:
                        raise ValueError('Unexpected class value in KMZ')
                    uncertainty[name.split('/')[1]] += values[UNKNOWN]
    if dict(uncertainty) != expected_by_tile:
        raise ValueError('Full-resolution KMZ uncertainty counts differ from prediction')
