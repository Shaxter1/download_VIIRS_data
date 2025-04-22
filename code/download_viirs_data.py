import subprocess
import os
import h5py
import xarray as xr
import numpy as np
import glob

CONCEPT_ID = "C1562021084-LAADS"
BOUNDING_BOX = "130.0,42.3,131.5,43.2"

def get_granules(concept_id, start_date, end_date, bounding_box):
    command = ['cmrfetch', 'granules', '-c', concept_id, '-t', f'{start_date},{end_date}', '--bounding-box', bounding_box]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')

        if result.stdout.strip():
            print("Найденные гранулы:")
            print(result.stdout)
            return result.stdout
        else:
            print(f"Не найдено гранул для концепта {concept_id} в указанный период и координатах.")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Ошибка выполнения команды: {e}")
    except FileNotFoundError:
        print("Команда cmrfetch не найдена. Убедитесь, что она установлена в PATH.")

    return None

def download_granules(concept_id, start_date, end_date, output_folder, bounding_box):
    command = ['cmrfetch', 'granules', '-c', concept_id, '-t', f'{start_date},{end_date}', '--bounding-box', bounding_box, '--download', output_folder]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')

        if result.returncode == 0:
            print(f"Данные для концепта {concept_id} загружены в папку {output_folder}.")
        else:
            print(f"Возможные проблемы при загрузке данных для {concept_id}. Проверьте файлы.")

    except subprocess.CalledProcessError as e:
        print(f"Ошибка загрузки данных: {e}")
    except FileNotFoundError:
        print("Команда cmrfetch не найдена. Убедитесь, что она установлена и доступна в PATH.")

def convert_to_xarray(folder):
    nc_files = glob.glob(os.path.join(folder, "*.nc"))

    if not nc_files:
        print("Не найдены файлы .nc для обработки.")
        return

    # Создаем подпапку для хранения xarray файлов
    xarray_folder = os.path.join(folder, "xarray.Dataset")
    os.makedirs(xarray_folder, exist_ok=True)

    for file_path in nc_files:
        try:
            with h5py.File(file_path, "r") as f:
                # Загружаем массивы
                latitude = f["geolocation_data/latitude"][:]  # [time, lat]
                longitude = f["geolocation_data/longitude"][:]  # [time, lon]
                time = f["scan_line_attributes/scan_start_time"][:]  # [time]
                cloud_mask = f["geophysical_data/Cloud_Mask"][:]  # [time, lat, lon]

                # Получаем реальные размеры
                time_dim, lat_dim, lon_dim = cloud_mask.shape

                # Усредняем широту и долготу по оси времени (если они 2D)
                latitude = latitude.mean(axis=0) if latitude.ndim == 2 else latitude
                longitude = longitude.mean(axis=0) if longitude.ndim == 2 else longitude

                # Если размер `latitude` не совпадает с `cloud_mask`
                if len(latitude) != lat_dim:
                    print(f"Интерполяция latitude: {len(latitude)} -> {lat_dim}")
                    lat_idx = np.linspace(0, len(latitude) - 1, lat_dim)
                    latitude = np.interp(lat_idx, np.arange(len(latitude)), latitude)

                # Если размер `longitude` не совпадает с `cloud_mask`
                if len(longitude) != lon_dim:
                    print(f"Интерполяция longitude: {len(longitude)} -> {lon_dim}")
                    lon_idx = np.linspace(0, len(longitude) - 1, lon_dim)
                    longitude = np.interp(lon_idx, np.arange(len(longitude)), longitude)

                # Проверяем размерность time
                if len(time) != time_dim:
                    time = np.linspace(time.min(), time.max(), time_dim)

                # Создаем Dataset
                ds = xr.Dataset(
                    {
                        "Cloud_Mask": (["time", "latitude", "longitude"], cloud_mask)
                    },
                    coords={
                        "latitude": (["latitude"], latitude),
                        "longitude": (["longitude"], longitude),
                        "time": (["time"], time),
                    }
                )

                print(f"Файл {file_path} успешно преобразован в xarray.Dataset")

                # Сохраняем Dataset в файл .nc
                output_file = os.path.join(xarray_folder, os.path.basename(file_path))
                ds.to_netcdf(output_file)
                print(f"xarray.Dataset сохранен в {output_file}")

        except Exception as e:
            print(f"Ошибка при обработке {file_path}: {e}")

def main():
    start_date = "2025-02-16T00:00:00Z"
    end_date = "2025-02-16T23:59:59Z"

    output_folder = os.path.join(os.environ.get('USERPROFILE', ''), 'Desktop', 'file_VIIRS')
    os.makedirs(output_folder, exist_ok=True)

    granules = get_granules(CONCEPT_ID, start_date, end_date, BOUNDING_BOX)

    if not granules:
        return

    download_granules(CONCEPT_ID, start_date, end_date, output_folder, BOUNDING_BOX)

    # После загрузки данных выполняем преобразование и сохранение
    convert_to_xarray(output_folder)

if __name__ == "__main__":
    main()
