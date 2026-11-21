#!/usr/bin/env python3
"""Read the local regional Cashew background for a country code."""

import argparse
import json
import re

from _runtime.common import SKILL_ROOT, read


def lookup_country_context(country):
    country = country.strip().upper()
    background = SKILL_ROOT / 'references/country-background.md'
    match = re.search(rf'^## {re.escape(country)}\n(.*?)(?=^## |\Z)',
                      background.read_text(encoding='utf-8'), re.MULTILINE | re.DOTALL)
    if match is None:
        return dict(country=country, status='no_local_summary')
    return dict(country=country, status='available', background=match.group(1).strip())


def administrative_area(country, bounds):
    """Locate the tile center in the project's public administrative boundaries."""
    from shapely.geometry import Point, shape

    boundaries = SKILL_ROOT.parents[2] / 'cashew_spatial_prior' / 'boundaries' / f'{country}-ADM1.geojson'
    if not boundaries.is_file():
        return None
    west, south, east, north = bounds
    center = Point((west + east) / 2, (south + north) / 2)
    return next((region['properties']['shapeName'] for region in read(boundaries)['features']
                 if shape(region['geometry']).covers(center)), None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('country', help='ISO 3166-1 alpha-3 code')
    args = parser.parse_args()
    print(json.dumps(lookup_country_context(args.country), ensure_ascii=False, indent=2))
