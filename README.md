# 🌿 Plant Classifier CNN

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

**Projekt zaliczeniowy** – Inteligentny system rozpoznawania gatunków roślin oraz diagnostyki chorób liści oparty na głębokich sieciach splotowych (CNN).

---

## 📖 Opis projektu
Aplikacja wykorzystuje uczenie maszynowe, aby pomóc użytkownikom w szybkiej identyfikacji roślin oraz monitorowaniu ich stanu zdrowia. System analizuje przesłane zdjęcie liścia i w czasie rzeczywistym zwraca informację o gatunku oraz ewentualnych jednostkach chorobowych.

### ✨ Główne funkcje
* **Klasyfikacja gatunku:** Automatyczne rozpoznawanie 9 popularnych rodzajów roślin.
* **Detekcja chorób:** Specjalistyczna analiza stanu zdrowia liści (obecnie zoptymalizowana dla jabłoni).
* **Intuicyjny interfejs:** Webowy panel użytkownika typu *drag-and-drop*.
* **Natychmiastowy wynik:** Przetwarzanie obrazu zajmuje zazwyczaj poniżej 2 sekund.

---

## 🛠️ Technologie
* **Deep Learning:** TensorFlow & Keras (Architektura CNN)
* **Backend:** Python
* **Frontend:** Streamlit
* **Przetwarzanie obrazu:** OpenCV

---

## 📂 Obsługiwane rośliny
Model został wytrenowany na zbiorze danych obejmującym następujące rośliny:
`Apple`, `Blueberry`, `Cherry`, `Grape`, `Pepper`, `Raspberry`, `Soybean`, `Strawberry`, `Tomato`.

## Technologie

- **Deep Learning:** TensorFlow/Keras (CNN)
- **Backend:** Python
- **Frontend:** Streamlit

## Screenshot

![Przykład klasyfikacji](example_classification.png)

---

## 🛠️ Planowane usprawnienia
Aby zwiększyć skuteczność modelu i zminimalizować błędy, planowane są następujące działania techniczne:

**Rozszerzenie zbioru danych treningowych:**
- Zaawansowana augmentacja danych: transformacje geometryczne (rotacja, odbicia lustrzane) oraz fotometryczne (zmiana jasności, kontrastu, nasycenia)
- Zróżnicowanie warunków oświetleniowych: zdjęcia o niskiej ekspozycji oraz przy świetle o różnej temperaturze barwowej, co zapobiega overfittingowi

**Implementacja Uczenia Aktywnego (Active Learning):**
- Mechanizm Human-in-the-loop: w przypadku niskiej pewności modelu (low confidence score) użytkownik wskazuje poprawną etykietę (Ground Truth)
- Zebrane dane posłużą do dotrenowania modelu w kolejnych iteracjach (Incremental Learning)

** 🏥 Diagnostyka i Zalecenia (System Ekspercki)
Aplikacja nie tylko rozpoznaje chorobę, ale również pełni rolę asystenta ogrodnika. Po wykryciu problemu system wyświetla dedykowaną sekcję z poradami:

** 🔍 Przyczyny wystąpienia chorób:
* **Czynniki środowiskowe:** Nadmierna wilgotność, brak cyrkulacji powietrza, zbyt wysokie zagęszczenie roślin.
* **Patogeny:** Informacje o grzybach, bakteriach lub wirusach odpowiedzialnych za dany stan (np. *Podosphaera leucotricha* w przypadku mączniaka jabłoni).
```

