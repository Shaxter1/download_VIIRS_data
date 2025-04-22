# 📡 VIIRS Data Downloader & Converter

Этот проект предназначен для автоматической загрузки, обработки и конвертации спутниковых данных VIIRS (Visible Infrared Imaging Radiometer Suite) с использованием утилиты `cmrfetch`. Скрипт выполняет поиск гранул, скачивание и преобразование данных в формат `xarray.Dataset`.

---

## Возможности

- Поиск гранул VIIRS по концепту (concept ID), дате и координатам
- Загрузка найденных `.nc` файлов в указанный каталог
- Преобразование HDF5 (`.nc`) файлов в `xarray.Dataset`
- Сохранение в формате NetCDF с новыми координатами

---

## Зависимости
- Python 3.8+
- [cmrfetch](https://github.com/nasa/CMR-CLI) (должен быть установлен и доступен в `PATH`)
- `h5py`
- `xarray`
- `numpy`
- `glob`

Установить зависимости:
```bash
pip install h5py xarray numpy
