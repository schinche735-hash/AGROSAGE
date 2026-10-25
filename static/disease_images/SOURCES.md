# Disease gallery image provenance

Files named `plantvillage_*.jpg` are small reference thumbnails generated from the local PlantVillage dataset using `datasets/create_reference_images.py`.

- Dataset catalogue: https://www.tensorflow.org/datasets/catalog/plant_village
- Dataset paper: Hughes & Salathé (2015), https://arxiv.org/abs/1511.08060
- Each original class label and relative source image path is recorded in `datasets/manifest.csv`.
- PlantVillage contains controlled-background images; these thumbnails are examples, not proof that a predicted diagnosis is correct.
- Verify the current dataset/image licence and preserve required attribution before redistributing the project or gallery images.

## Soybean, wheat, and tur/pigeonpea reference thumbnails
These ten thumbnails are real examples selected from the same labelled datasets used to train the expanded classifier. They are for visual reference in the disease knowledge library; they are not synthetic images.

- **Soybean** (Bacterial leaf blight, dry leaf, Septoria brown spot, vein necrosis): Project-AgML, *Soybean Leaf Disease Classification*, CC BY 4.0. Dataset and citation: https://huggingface.co/datasets/Project-AgML/soybean_leaf_disease_classification (Kotwal, Jameer; Kashyap, Ramgopal; Pathan, Mohd Shafi, 2024, Data in Brief 53, 110216; source dataset DOI 10.17632/bshkvgbzpt.1).
- **Tur / pigeonpea** (Cercospora leaf spot, sterility mosaic disease): G. G. Rajput and Vanita Doddamani, *A novel pigeonpea leaf dataset for detection and classification of pigeonpea leaf diseases*, Mendeley Data, DOI: 10.17632/bd553pdtny.1, CC BY 4.0. https://data.mendeley.com/datasets/bd553pdtny/1
- **Wheat** (brown rust, powdery mildew, Septoria, yellow rust): Megan C. Long, Matthew Hartley, Richard J. Morris, and James K. M. Brown, *Wheat disease images (small dataset)*, Zenodo, DOI: 10.5281/zenodo.7573133. https://zenodo.org/records/7573133. The record does not clearly expose a reuse license in its metadata; confirm the dataset terms before public redistribution of these wheat thumbnails.

Thumbnail filenames are generated from the labelled source folders and use the `plantvillage_` filename prefix only for compatibility with the app's existing image lookup; these are **not PlantVillage images**.
