# Cashew climate and cocoa confounders — regional context, not pixel truth

Version2026-11-15; prepared for the CIV/GHA sample-assist study using public primary/official sources. No target reference labels, annotations, accuracy or prediction maps were used to write this note. Use when regional context is relevant; configured tools may retrieve more local evidence. Record source scope/year. These are agronomic and regional hypotheses, not universal classifier thresholds.

## Cashew climatic support

Cashew is a tropical crop intolerant of frost. UNCTAD describes approximately26–28°C mean monthly temperatures as optimal and roughly1,000–2,000mm rain with a5–7month rainy season and a pronounced dry period for flowering/fruit set as favourable conditions. These are indicative favourable ranges, not absolute survival or presence limits; cultivars, soils, management and irrigation matter. Evaluate seasonal distribution, not annualrainfall alone. Source: UNCTAD2021, Commodities at a Glance—Special issue on cashew nuts, §2.1, https://unctad.org/system/files/official-document/ditccom2020d1_en.pdf .

Cashew requires drainage; waterlogging/saline or alkaline sites can be unsuitable, and apparent floodplain/wetland trees should trigger local evidence checks rather than being labelled Cashew because woody texture resembles a plantation. Some canopy rows/soil exposure reflect management, not species identity. Source: FAO integrated production practices inIndia, §§4.1,5.2, https://www.fao.org/4/ac451e/ac451e04.htm .

Drought tolerance is not proof that only dry landscapes can grow Cashew. FAO'sThailand account describes production in comparatively humid areas and the importance of available rootzone moisture with gooddrainage: https://www.fao.org/4/ac451e/ac451e09.htm . Do not impose a hard north/south or moist/dry exclusion.

For a target, use measured monthly climate context when helpful: mean temperature, rainseason concentration, duration/timing of drymonths and flood/drainage evidence. A day's weather is not climatic suitability. NASA POWER get_climate_context returns coarse climatology with its real baseline period and units. Do not pretend those grids resolve10m or represent rainfall of the imagecapture year. Suitableclimate increases plausibility, not observed crop identity.

## Ghana: cashew/cocoa geography

Ghana'sTreeCropsDevelopmentAuthority identifies Bono, BonoEast and Ahafo as major Cashew production regions. This is a recent sector statement and not a historical fieldboundary map: https://tcda.gov.gh/tcda-boss-dr-andy-osei-okrah-pushes-for-value-addition-in-cashew-apple-to-boost-economic-growth/ .

MoFA records Cashew cultivation/processing around Wenchi, Tain, JamanNorth/South and Techiman in the historicalBrongAhafo administrative area: https://mofa.gov.gh/site/directorates/45-regional-directorates/brong-ahafo-region/79-programmes-projects-potentials-ba . Administrative boundaries changed; match modern location names carefully. These examples are not an exhaustive plantation inventory.

COCOBOD'ssector geography includes WesternNorth, WesternSouth, Ashanti, BrongAhafo, Eastern, Central andVolta production/operational areas: https://cocobod.gh/cocoa-story and https://cocobod.gh/cocoa-purchases . These sector regions are not always the current16administrative regions. In particular the oldBrongAhafo belt includes settings where cocoa and Cashew require explicit local comparison. Never infer 'Bono meansCashew' or 'cocoa region means noCashew'.

## Côte d'Ivoire: production context

The government'sfirstBiennialTransparencyReport locates cocoa/coffee principally in the southern/southwestern forest zone, including Soubré, Daloa andSanPédro; northern savanna production includes Cashew/cotton, including Korhogo, Boundiali andFerkessédougou. Source: government submission toUNFCCC, agricultural production zones section, https://unfccc.int/sites/default/files/resource/Premier%20rapport%20Biennal%20sur%20la%20Transparence%20C%C3%B4te%20d%27Ivoire.pdf . These locations are examples, not cropboundary exclusions or pointlabels.

Coffee-CocoaBoard'spublic offices include Abengourou, Aboisso, Adzopé/Agboville, Bongouanou, Daloa, Duékoué, Gagnoa, Man, SanPédro, Soubré andYamoussoukro: https://conseilcafecacao.ci/index.php?id=77%3Acoffee-cocoa-board&option=com_k2&print=1&tmpl=component&view=item . Office locations indicate a sector presence, not measured cocoa extent around an AOI. Public2022/2023 regional cocoa/coffee production data are available at https://data.gouv.ci/datasets/repartition-de-la-production-de-cafe-et-cacao-par-region-administrative-en-tonnes .

## Cocoa as an explicit lookalike hypothesis

The user identifies cocoa plantations as an important Cashew confounder. COCOBOD/UNDP documents cocoa landscapes in Ghana'shumidforest zones and the integration of shade/other trees: https://www.undp.org/ghana/projects/environmentally-sustainable-production-practices-cocoa-landscapes . Cocoa can therefore appear as dense woody vegetation or agroforestry; canopyshade can obscure the understory crop. Managementdiversity prevents a universal 'dense/dark canopy = cocoa' rule.

The following are interpretation checks, NOT validated RGBspecies rules: compare stand texture, canopy-height layering/shadetrees, crownspacing/sizevariation, interrowopenness, parcelcontinuity and seasonalappearance against the actual SOURCE TRAIN examples. Request samecentre detail from bothproviders or another date when needed. Consider cocoa, otherorchards, naturalwoodland and mixedagroforestry as alternative hypotheses. Regional/climate support should change what the Agent investigates; it cannot substitute for local evidence or force a10m class.

If cocoa/nonCashew is supported but its four-class subtype is unresolved, directbinary may be0 while fourclass is10. If Cashewversuscocoa itself is unresolved, both relevant decisions remain10. Don't teach the model that every sourceclass1 includes cocoa, or assign allcroptrees toCashew. The regionID is geometryonly.

## 10m land-unit interpretation

SAM supplies geometry and Sol decides class using the actual RGB and references. Normal orchard inter-row ground and an incidental tree in a crop field follow the dominant land-unit meaning. Unresolved crop identity and genuine cross-land-use transitions remain 10. Revise or split masks that join unlike parcels. Use regional suitability to contextualize the visual evidence.
