import kagglehub

print("=" * 60)
print("DOWNLOADING EXTERNAL BENCHMARK DATASETS")
print("=" * 60)

# =========================================================
# FaceForensics++
# =========================================================

faceforensics_path = kagglehub.dataset_download(
    "greatgamedota/faceforensics"
)

print("\nFaceForensics++ downloaded:")
print(faceforensics_path)

# =========================================================
# Celeb-DF
# =========================================================

celebdf_path = kagglehub.dataset_download(
    "reubensuju/celeb-df-v2"
)

print("\nCeleb-DF downloaded:")
print(celebdf_path)

# =========================================================
# DFDC
# =========================================================

dfdc_path = kagglehub.dataset_download(
    "xhlulu/dfdc-preview"
)

print("\nDFDC downloaded:")
print(dfdc_path)

print("\nDONE.")