# Tunisia Parking/ALPR Dataset (Local)

Ce dossier contient des données tunisiennes de départ pour SmartPark :

- `vehicle_registry_tn.csv` : registre plaques locales + catégories d'accès
- `parking_policy_tn.json` : règles tarifaires et accès conditionnel
- `alpr_tn_annotations.csv` : annotations ALPR (bbox + conditions capture)

## Utilisation

- Le moteur de règles charge automatiquement `vehicle_registry_tn.csv` et `parking_policy_tn.json`.
- `alpr_tn_annotations.csv` sert de base de vérité terrain pour calibration/évaluation OCR et détection.

## Remarque

Ce dataset est un jeu de données initial/synthétique pour prototypage. Pour production, remplacer par des captures réelles annotées (respect RGPD/local policy).
