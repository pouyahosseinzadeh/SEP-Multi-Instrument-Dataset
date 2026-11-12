#!/usr/bin/env python3
"""Reproduce native-data response fits and report failed source bridges.

This developer/reproducibility command retrieves only the listed public NOAA
calibration days. Normal dataset repair uses the already packaged model bank.
Every fit is trained and tested on distinct whole days. Failed fits are saved
for inspection and are never inserted into the usable model bank.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
from urllib.parse import urljoin
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sep_dataset.proton_calibration import (
    BANK_SCHEMA, FEATURES, KEY_FIELDS, CalibrationError, apply_calibration,
    fit_calibration, fit_official_2017, frame_source_key, load_calibration,
    save_calibration, validation_table,
)
from sep_dataset.sgps_sources import ARCHIVE, SPECIAL_2017, read_l1b, read_l2


LEGACY = 'https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/2017/09/goes15/csv/g15_epead_p17ew_5m_20170901_20170930.csv'
# (day, split, regime); no random five-minute interleaving.
EARLY = [('2020-11-29','train','quiet'), ('2021-10-28','train','event'),
         ('2021-11-02','train','quiet'), ('2022-03-28','train','event'),
         ('2021-10-29','validation','event'), ('2021-11-01','validation','quiet'),
         ('2022-03-29','validation','event')]
LATE = [('2024-01-01','train','quiet'), ('2024-05-11','train','event'),
        ('2024-09-14','train','event'), ('2024-10-09','train','event'),
        ('2024-01-02','validation','quiet'), ('2024-05-12','validation','event'),
        ('2024-10-10','validation','event'), ('2025-01-02','validation','quiet'),
        ('2025-02-24','validation','event')]
G19 = [('2025-07-01','train','quiet'), ('2025-02-24','train','event'),
       ('2025-05-14','train','event'), ('2025-11-10','train','event'),
       ('2025-07-02','validation','quiet'), ('2025-02-25','validation','event'),
       ('2025-11-11','validation','event'), ('2025-11-12','validation','event')]
PLANS = {'G16L1b': (16, 'L1b', EARLY[:3] + EARLY[4:6] + LATE[1:4] + LATE[4:7]),
         'G17L2': (17, 'L2', EARLY), 'G18L2': (18, 'L2', LATE),
         'G19L2': (19, 'L2', G19)}


def fetch(url, path):
    if path.exists():
        return path
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix+'.{}.part'.format(threading.get_ident()))
    part.write_bytes(content)
    part.replace(path)
    return path


def find_product(cache, satellite, level, day):
    day = pd.Timestamp(day)
    folder = f'l2/data/sgps-l2-avg5m/{day:%Y/%m}/' if level == 'L2' else f'l1b/seis-l1b-sgps/{day:%Y/%m}/'
    directory = f'{ARCHIVE}/goes{satellite}/'+folder
    listing = cache/f'index_g{satellite}_{level}_{day:%Y%m}.html'
    fetch(directory, listing)
    links = re.findall(r'href=[\"\']([^\"\']+)[\"\']', listing.read_text())
    names = [urljoin(directory, link) for link in links if f'_g{satellite}_d{day:%Y%m%d}_' in link and link.endswith('.nc')]
    if not names:
        raise CalibrationError(f'No {level} product for GOES-{satellite} {day:%Y-%m-%d}')
    def version(url):
        match = re.search(r'_v([\d_-]+)\.nc$', url)
        return tuple(int(v) for v in re.split('[-_]',match[1])) if match else ()
    url = sorted(names, key=lambda x:(version(x), '/sci_' in x), reverse=True)[0]
    path = cache/Path(url).name
    fetch(url, path)
    return path, url


def native_frame(path, satellite, level):
    frame, details = (read_l2 if level == 'L2' else read_l1b)(path, satellite)
    return frame, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', default='/tmp/sep-bridge-data')
    parser.add_argument('--outdir', default=str(Path(__file__).parent))
    parser.add_argument('--source', action='append', choices=list(PLANS))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    cache, output = Path(args.cache_dir), Path(args.outdir)
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    reference_path = output/'goes15_reference_response_v1.json'
    if reference_path.exists():
        reference = load_calibration(reference_path)
    else:
        native = fetch(SPECIAL_2017,cache/'sgps_201709.nc')
        legacy = fetch(LEGACY,cache/'legacy_g15_201709.csv')
        reference = fit_official_2017(native,legacy,provenance={'sgps_url':SPECIAL_2017,'legacy_url':LEGACY})
        save_calibration(reference_path,reference)
    validation_table(reference).to_csv(output/'goes15_reference_validation_v1.csv',index=False)
    requested = args.source or list(PLANS)
    jobs = {}
    for name in requested:
        sat,level,plan = PLANS[name]
        reference_satellite = 18 if name == 'G19L2' else 16
        for day,_,_ in plan:
            jobs[(reference_satellite,'L2',day)] = None
            jobs[(sat,level,day)] = None
    errors, files = {}, {}
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures = {pool.submit(find_product,cache,*key):key for key in jobs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                files[key] = future.result()
                print('[SOURCE]',key,files[key][0].stat().st_size,flush=True)
            except Exception as exc:
                errors[key] = str(exc)
                print('[UNAVAILABLE]',key,exc,flush=True)
    # netCDF/HDF5 reads are deliberately sequential; their underlying libraries
    # need not be thread safe even though HTTP downloads may run concurrently.
    parsed = {}
    for key,(path,url) in files.items():
        try:
            frame,details = native_frame(path,key[0],key[1])
            parsed[key] = (frame,details)
            print('[PARSED]',key,len(frame),frame.attrs.get('native_valid_points'),flush=True)
        except Exception as exc:
            errors[key] = str(exc)
            print('[INVALID]',key,exc,flush=True)
    bank_path = output/'proton_calibration_bank.json'
    if bank_path.exists():
        existing = json.loads(bank_path.read_text())
        artifacts = [reference]
        artifacts.extend(a for a in existing.get('artifacts',[]) if a.get('provenance',{}).get('bridge_plan') not in requested and a.get('reference_kind')!='observed_legacy' and reference['calibration_id'] in a.get('lineage',[]))
    else:
        artifacts = [reference]
    if reference['calibration_id'] not in {a['calibration_id'] for a in artifacts}:
        artifacts.insert(0,reference)
    coverage = [entry for entry in (existing.get('coverage',[]) if bank_path.exists() else []) if entry.get('bridge_plan') not in requested]
    for name in requested:
        satellite,level,plan = PLANS[name]
        reference_satellite = 18 if name == 'G19L2' else 16
        reference_model = reference
        if reference_satellite != 16:
            possible = [a for a in artifacts if a['applicability']['source_key']['source_satellite']==f'GOES-{reference_satellite}' and a['applicability']['source_key']['source_product']=='NOAA_SGPS_L2_AVG5M']
            if len(possible) != 1:
                coverage.append({'bridge_plan':name,'status':'failed','reason':'Required validated G18 L2 parent model is unavailable'})
                continue
            reference_model = possible[0]
        pieces,targets,source_records = [],[],[]
        unavailable = []
        for day,split,regime in plan:
            a,b = (satellite,level,day),(reference_satellite,'L2',day)
            if a not in parsed or b not in parsed:
                unavailable.append({'day':day,'split':split,'reason':errors.get(a) or errors.get(b) or 'not parsed'})
                continue
            source,details = parsed[a]
            reference_native,_ = parsed[b]
            try:
                reference_values = apply_calibration(reference_native,reference_model)
            except CalibrationError as exc:
                unavailable.append({'day':day,'split':split,'reason':str(exc)})
                continue
            source = source.copy()
            source['calibration_split'] = split
            source['calibration_regime'] = regime
            source['calibration_day'] = day
            pieces.append(source)
            targets.append(reference_values[list(FEATURES)].reindex(source.index))
            for key in (a,b):
                path,url=files[key]
                source_records.append({'satellite':key[0],'level':key[1],'day':day,'url':url,
                                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        if not pieces:
            coverage.append({'bridge_plan':name,'status':'unavailable','unavailable_days':unavailable})
            continue
        source = pd.concat(pieces).sort_index()
        target = pd.concat(targets).reindex(source.index)
        if source.index.has_duplicates:
            raise CalibrationError('Overlapping calibration days')
        for group_values,group in source.groupby(list(KEY_FIELDS),dropna=False,sort=False):
            key=dict(zip(KEY_FIELDS,map(str,group_values)))
            target_group=target.reindex(group.index)
            full_train=group.calibration_split.eq('train').to_numpy()
            full_validation=group.calibration_split.eq('validation').to_numpy()
            parent_supported=target_group[list(FEATURES)].notna().all(axis=1).to_numpy()
            train=full_train & parent_supported
            validation=full_validation & parent_supported
            quiet=validation & group.calibration_regime.eq('quiet').to_numpy()
            event=validation & group.calibration_regime.eq('event').to_numpy()
            upstream_coverage={}
            for regime in ('quiet','event'):
                all_rows=full_validation & group.calibration_regime.eq(regime).to_numpy()
                upstream_coverage[regime]={'requested_rows':int(all_rows.sum()),
                    'reference_supported_rows':int((all_rows & parent_supported).sum()),
                    'reference_coverage':float((all_rows & parent_supported).sum()/all_rows.sum()) if all_rows.any() else 0.0}
            label = name+'_'+hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:8]
            try:
                fit=fit_calibration(group,target_group,train,validation,
                                    validation_regimes={'quiet':quiet,'event':event},
                                    reference_kind='calibrated_reference_bridge',parent_artifact=reference_model,
                                    provenance={'bridge_plan':name,'planned_days':plan,'native_sources':source_records,
                                                'unavailable_days':unavailable,'target_definition':f'Same-time passed G{reference_satellite} L2 reference response estimates',
                                                'upstream_reference_coverage':upstream_coverage,
                                                'coverage_condition':'Bridge validation requested rows are fixed before fit as times with valid simultaneous parent-reference estimates; candidate missing predictions still count against the 95% requirement. Full-pool parent coverage is reported separately.',
                                                'scope_note':'This is an empirically validated bridge to modeled legacy-equivalent values, not direct validation against a live legacy instrument.'})
                destination=output/('bridges' if fit['status']=='passed' else 'failed_fit_reports')
                destination.mkdir(exist_ok=True)
                save_calibration(destination/(label+'.json'),fit)
                validation_table(fit).to_csv(destination/(label+'_validation.csv'),index=False)
                if fit['status']=='passed':
                    artifacts.append(fit)
                coverage.append({'bridge_plan':name,'source_key':key,'status':fit['status'],
                                 'calibration_id':fit['calibration_id'],'failure_reasons':fit['failure_reasons'],
                                 'planned_days':len(plan),'unavailable_days':unavailable,'upstream_reference_coverage':upstream_coverage,
                                 'validation':{f:v['validation'] for f,v in fit['models'].items()}})
                print('[FIT]',label,fit['status'],fit['failure_reasons'],flush=True)
            except Exception as exc:
                coverage.append({'bridge_plan':name,'source_key':key,'status':'failed','reason':str(exc),'unavailable_days':unavailable})
                print('[FIT FAILED]',label,exc,flush=True)
    bank={'schema':BANK_SCHEMA,'artifacts':artifacts,'coverage':coverage,
          'limitations':['Legacy-equivalent estimates, not native EPEAD measurements',
                         'Bridges inherit reference-model errors and are validated against modeled targets',
                         'Only passed source keys, compatible numerical energy grids and tested spectra may be used',
                         'Missing or failed source models are not replaced by mean equalization or another satellite model']}
    part=bank_path.with_suffix('.json.part')
    part.write_text(json.dumps(bank,indent=2,allow_nan=False)+'\n')
    part.replace(bank_path)
    (output/'bridge_collection_report.json').write_text(json.dumps({'requested_sources':requested,
        'downloaded_files':len(files),'unavailable_sources':[{'source':str(k),'reason':v} for k,v in errors.items()],
        'coverage':coverage},indent=2,allow_nan=False)+'\n')
    print('[BANK]',len(artifacts),'validated artifacts',bank_path,flush=True)


if __name__=='__main__':
    main()
