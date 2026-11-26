"""
Série temporelle Sentinel-1 RTC (gamma0), composites estivaux médians par
année, sur une même trace orbitale : même orbite relative -> angle
d'incidence quasi constant d'une année à l'autre, condition nécessaire pour
attribuer une variation inter-annuelle de rétrodiffusion à un changement de
couvert plutôt qu'à un changement de géométrie de visée. Source : catalogue
STAC public Microsoft Planetary Computer (gratuit, sans compte ni jeton).

La recherche, le choix de l'orbite commune et le composite médian viennent du paquet nisar_s1
(nisar_s1.cband), partagé avec les études NISAR / Sentinel-1.

Utilisation :
    TBE_AOI=<nom_aoi> python3 scripts/os1_download_data.py
"""

import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import box, shape

from nisar_s1.cband import common_orbit, rtc_composite, search_s1_rtc, signed_href
from nisar_s1.grid import make_grid

from config import AOI_BBOX, NODATA, RAW_DIR, RESOLUTION, SEASON_END_MD, SEASON_START_MD, YEARS

AOI_POLY = box(*AOI_BBOX)


def search_year(year):
    """Sentinel-1 RTC scenes of the seasonal window of ``year`` that intersect the AOI."""
    df = search_s1_rtc(AOI_BBOX, f"{year}-{SEASON_START_MD}", f"{year}-{SEASON_END_MD}")
    if len(df):
        df = df[[shape(it.geometry).intersects(AOI_POLY) for it in df.item]].reset_index(drop=True)
    return df


def target_grid(ref_item, resolution=RESOLUTION):
    """Grid of the study: AOI box corners projected to the CRS of the first scene of the common
    orbit, starting at its upper-left corner (not snapped)."""
    with rasterio.open(signed_href(ref_item.assets["vv"].href)) as ds:
        crs = ds.crs
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tr.transform([AOI_BBOX[0], AOI_BBOX[2]], [AOI_BBOX[1], AOI_BBOX[3]])
    return make_grid((min(xs), min(ys), max(xs), max(ys)), resolution, str(crs), snap=False)


def save(path, r):
    a = np.where(np.isnan(r.data), NODATA, r.data).astype(np.float32)
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1, "height": a.shape[0], "width": a.shape[1],
        "crs": r.crs, "transform": r.transform, "nodata": NODATA, "compress": "LZW",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(a, 1)


def main():
    print("Recherche des scènes Sentinel-1 RTC par année (Planetary Computer)...")
    by_year = {y: search_year(y) for y in YEARS}
    for y, df in by_year.items():
        print(f"  {y}: {len(df)} scène(s) trouvée(s)")
        if not len(df):
            raise SystemExit(
                f"Aucune scène en {y} sur cette fenêtre/AOI - élargir "
                f"SEASON_START_MD/SEASON_END_MD dans config.py ou vérifier l'AOI."
            )

    orbit = common_orbit(by_year)
    n_years = sum((df.relative_orbit == orbit).any() for df in by_year.values())
    print(f"Orbite relative retenue : {orbit} (disponible {n_years}/{len(by_year)} années)")

    ref_item = next(it for y in YEARS for it in by_year[y].item if it.properties.get("sat:relative_orbit") == orbit)
    grid = target_grid(ref_item)
    h, w = grid.data.shape
    print(f"Grille cible : {w}x{h} px @ {RESOLUTION:.0f} m ({grid.crs})")

    for k, y in enumerate(YEARS, 1):
        df = by_year[y]
        same = df[df.relative_orbit == orbit]
        use = same if len(same) else df
        if not len(same):
            print(f"  [!] {y}: aucune scène sur l'orbite {orbit}, repli sur "
                  f"{len(use)} scène(s) d'une autre orbite (à signaler dans les résultats)")
        print(f"  [{k}/{len(YEARS)}] {y}: composite médian sur {len(use)} scène(s) ({', '.join(use.id)})")
        for pol in ("vv", "vh"):
            comp = rtc_composite(list(use.item), pol, grid, stat="median", resampling="bilinear")
            out = RAW_DIR / f"s1_{y}_{pol}.tif"
            save(out, comp)
            print(f"    {out.name}  (valides: {np.isfinite(comp.data).mean() * 100:.0f}%)")

    print(f"\nTerminé. Données dans : {RAW_DIR}")


if __name__ == "__main__":
    main()
