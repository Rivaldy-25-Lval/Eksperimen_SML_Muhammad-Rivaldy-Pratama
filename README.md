# Eksperimen Dataset Wine Quality - Muhammad Rivaldy Pratama

Repository untuk submission Dicoding - Membangun Sistem Machine Learning (Kriteria 1 - ADVANCE)

## Struktur Folder

```
Eksperimen_SML_Muhammad-Rivaldy-Pratama/
├── .github/workflows/
│   └── preprocessing.yml          # GitHub Actions workflow
├── Eksperimen_Muhammad_Rivaldy_Pratama.ipynb  # Notebook eksperimen
└── preprocessing/
    ├── automate_Muhammad_Rivaldy_Pratama.py   # Script otomasi preprocessing
    └── data/preprocessed/
        ├── train_data.csv         # Data training (788 samples)
        ├── test_data.csv          # Data testing (197 samples)
        └── scaler.pkl             # StandardScaler object
```

## Kriteria yang Dipenuhi

✅ **Kriteria 1: ADVANCE (4 pts)**
- Eksperimen manual di notebook dengan EDA lengkap
- File otomasi preprocessing yang dapat dijalankan ulang
- GitHub Actions workflow untuk preprocessing otomatis

## Dataset

- **Source**: Wine Quality Dataset (Red Wine)
- **Original**: 1599 samples, 11 features
- **After cleaning**: 985 samples (removed 240 duplicates, 374 outliers)
- **Split**: 788 training / 197 testing (80/20)
- **Classes**: 3 (Low, Medium, High)

## Cara Menjalankan

### 1. Manual Preprocessing
```bash
python preprocessing/automate_Muhammad_Rivaldy_Pratama.py
```

### 2. Otomatis via GitHub Actions
- Push ke branch main
- Workflow akan otomatis menjalankan preprocessing
- Hasil tersimpan di artifacts

## Author

Muhammad Rivaldy Pratama
- GitHub: [@Rivaldy-25-Lval](https://github.com/Rivaldy-25-Lval)
- Dicoding: Rivaldy-25-Lval

## License

Submission project untuk Dicoding - Membangun Sistem Machine Learning
