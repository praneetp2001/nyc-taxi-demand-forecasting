"""Download NYC TLC taxi zone lookup + shapefile, compute centroids and H3 cells."""
import sys
import zipfile

import geopandas as gpd
import h3
import pandas as pd
import requests

from src.config import DATA_RAW, DATA_PROCESSED, ZONE_LOOKUP_URL, ZONE_SHAPEFILE_URL, H3_RESOLUTION


def download(url: str, dest_path) -> None:
    if dest_path.exists():
        print(f"[skip] {dest_path.name} already downloaded")
        return
    print(f"[download] {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest_path.write_bytes(r.content)
    print(f"[done] {dest_path.name} ({len(r.content) / 1e3:.0f} KB)")


def main() -> None:
    lookup_path = DATA_RAW / "taxi_zone_lookup.csv"
    shapefile_zip = DATA_RAW / "taxi_zones.zip"
    shapefile_dir = DATA_RAW / "taxi_zones"

    download(ZONE_LOOKUP_URL, lookup_path)
    download(ZONE_SHAPEFILE_URL, shapefile_zip)

    if not shapefile_dir.exists():
        with zipfile.ZipFile(shapefile_zip) as zf:
            zf.extractall(shapefile_dir)
        print(f"[unzip] -> {shapefile_dir}")

    shp_files = list(shapefile_dir.glob("**/*.shp"))
    gdf = gpd.read_file(shp_files[0])

    # Centroid in the shapefile's native projected CRS (feet-based), then reproject to lat/lon.
    centroids_proj = gdf.geometry.centroid
    centroids_ll = gpd.GeoSeries(centroids_proj, crs=gdf.crs).to_crs(epsg=4326)

    zones = pd.DataFrame(
        {
            "LocationID": gdf["LocationID"].astype(int),
            "centroid_lat": centroids_ll.y,
            "centroid_lon": centroids_ll.x,
        }
    )
    zones["h3_cell"] = zones.apply(
        lambda row: h3.latlng_to_cell(row["centroid_lat"], row["centroid_lon"], H3_RESOLUTION),
        axis=1,
    )

    lookup = pd.read_csv(lookup_path)
    zones = zones.merge(lookup, on="LocationID", how="left")

    out_path = DATA_PROCESSED / "zones.parquet"
    zones.to_parquet(out_path, index=False)
    print(f"[saved] {out_path} shape={zones.shape}")
    print(zones.head())


if __name__ == "__main__":
    sys.exit(main())
