# Stage 3: Dataset Loading and Splitting

## Objective

The objective of this stage was to load the preprocessed Adinkra image dataset into PyTorch and prepare it for training a deep learning model.

The dataset loader is responsible for:

- Loading the preprocessed images.
- Automatically detecting the 101 Adinkra symbol classes.
- Assigning numerical labels to each class.
- Applying image transformations.
- Applying data augmentation to training images.
- Splitting the dataset into training, validation, and testing sets.
- Creating PyTorch DataLoaders for batch processing.
- Detecting available hardware and moving batches to the available device.

---

## Dataset Loading

The dataset loader reads the preprocessed images from:

`dataset/processed`

The images were previously processed to a standard size of **224 × 224 pixels**.

PyTorch's `ImageFolder` dataset was used to automatically:

1. Identify the class folders.
2. Load the images.
3. Assign each class a numerical label.
4. Return images and their corresponding labels.

### Dataset Statistics

| Property | Value |
| -------- | ----: |
| Total Images | 5,089 |
| Number of Classes | 101 |
| Image Size | 224 × 224 |
| Image Channels | 3 (RGB) |
| Batch Size | 32 |

---

## Classes

The current dataset contains **101 Adinkra symbol classes**:

1. Aban
2. Abe_Dua
3. Adikrahene_Dua
4. Adinkrahene
5. Adwera
6. Adwo
7. Agyinduwura
8. Akoben
9. Akofena
10. Akokonan
11. Akoma
12. Akoma_Ntoso
13. Ananse_Ntontan
14. Ani_Bere
15. Asase_Ye_Duru
16. Aya
17. Bese_Saka
18. Bi_Nnka_Bi
19. Biribi_Wo_Soro
20. Boa_Me_Na_Me_Boa_Wo
21. Dame_Dame
22. Denkyem
23. Dono
24. Duafe
25. Dwannimmen
26. Eban
27. Epa
28. Ese_Ne_Tekrema
29. Fafanto
30. Fawohudie
31. Fihankra
32. Fofo
33. Funtumfunafu_Denkyem_Funafu
34. Gye_Nyame
35. Hwemudua
36. Hye_Won_Hye
37. Kae_Me
38. Kete_Pa
39. Kintinkantan
40. Kojo_Baiden
41. Kontire_Ne_Akwamu
42. Krado
43. Kramo_Bone
44. Kuntinkantan
45. Kwatakye_Atiko
46. Mako
47. Mate_Masie
48. Mframadan
49. Mmere_Dane
50. Mmomudwan
51. Mmusuyidee
52. Mpatapo
53. Mpuannum
54. Nea_Onnim_No_Sua_A_Ohu
55. Nea_Ope_Se_Nkrofoo_Ye_Ma_Wo_No-_Ye_Saa_Ara_Ma_won
56. Nea_Ope_Se_Obedi_Hene
57. Nkonsonkonson
58. Nkontim
59. Nkuma_Kese
60. Nkyimu
61. Nkyinkyim
62. Nnonnowa
63. Nsaa
64. Nserewa
65. Nsoromma
66. Nya_Abotere
67. Nyame_Akruma
68. Nyame_Biribi_Wo_Soro
69. Nyame_Dua
70. Nyame_Nnwu_Na_Mawu
71. Nyame_Nti
72. Nyame_Ye_Ohene
73. Nyansapo
74. Odo_Nyera_Fie_Kwan
75. Ohen_Adwae
76. Ohene
77. Ohene_Aniwa
78. Ohene_Tuo
79. Okodee_Mmowere
80. Okuafo_Pa
81. Onyakopon_Adom_Nti_Biribiara_Beye_Yie_African_Adinkra_Weddin
82. Onyakopon_Aniwa
83. Onyakopon_Ne_Yen_Ntena
84. Osidan
85. Osram
86. Osram_Ne_Nsoromma
87. Owo_Foro_Adobe
88. Owuo_Atwedee
89. Owuo_Kum_Nyame
90. Pa_Gya
91. Sankofa
92. Sepow
93. Sesa_Woruban
94. Sunsum
95. Tabon
96. Tamfo_Bebre
97. Tumi_Te_Se_Kosua
98. Tuo_Ne_Akofena
99. Wawa_Aba
100. Wo_Nsa_Da_Mu_A
101. Wuforo_Dua_Pa_A

---

## Class Mapping

PyTorch's `ImageFolder` automatically assigns numerical labels to the classes in alphabetical order.

Examples from the generated mapping:

| Class | Label |
| ----- | ----: |
| Aban | 0 |
| Abe_Dua | 1 |
| Adikrahene_Dua | 2 |
| Adinkrahene | 3 |
| Adwera | 4 |
| Adwo | 5 |
| Agyinduwura | 6 |
| Akoben | 7 |
| Akofena | 8 |
| Akokonan | 9 |
| ... | ... |
| Sankofa | 90 |
| Sepow | 91 |
| Sesa_Woruban | 92 |
| Sunsum | 93 |
| Tabon | 94 |
| Tamfo_Bebre | 95 |
| Tumi_Te_Se_Kosua | 96 |
| Tuo_Ne_Akofena | 97 |
| Wawa_Aba | 98 |
| Wo_Nsa_Da_Mu_A | 99 |
| Wuforo_Dua_Pa_A | 100 |

Therefore, the dataset contains labels from **0 through 100**, representing the 101 classes.

---

## Data Transformations

The dataset loader applies different transformations depending on whether the image is being used for training or evaluation.

### Training Transformations

Data augmentation is applied **only to the training dataset**.

The following transformations are used:

- Resize to 224 × 224 pixels.
- Random rotation of up to 15 degrees.
- Random horizontal flipping.
- Random cropping with padding.
- Conversion to PyTorch tensors.
- Image normalization.

The training transformation pipeline is:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
Stage 3 was successfully completed.

The 5,089 preprocessed Adinkra images have been loaded into PyTorch and organized into 101 classes. The dataset was split reproducibly into training, validation, and testing sets using a 70/15/15 ratio.

Training images receive data augmentation to help the CNN generalize to variations in image appearance, while validation and testing images remain unaugmented for consistent evaluation.

The images are converted into PyTorch tensors with dimensions of 3 × 224 × 224, and DataLoaders prepare the data in batches of 32 images.

The pipeline also includes automatic CUDA detection. Although the current development machine does not have an NVIDIA CUDA-enabled GPU and therefore runs on CPU, the code is prepared to use GPU acceleration when a CUDA-enabled environment is available.

Dataset prepared and loaded into PyTorch tensors and batches. The data pipeline is ready for CNN model development and training.