import argparse
import json
import math
import re
import csv
from statistics import mean, median
from pathlib import Path
from datetime import datetime

# Reuse same extractor + haversine logic
float_re = re.compile(r'(-?\d+\.\d+)')

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def extract_latlon(text):
    if not text:
        return None
    nums = float_re.findall(text)
    if len(nums) >= 2:
        try:
            lat = float(nums[0])
            lon = float(nums[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except Exception:
            pass
    txt = text.replace('\n',' ').lower()
    m = re.search(r'lat(?:itude)?[:=\s]+(-?\d+\.\d+).{0,40}lon(?:gitude)?[:=\s]+(-?\d+\.\d+)', txt)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except Exception:
            pass
    return None

def analyze_files(input_paths):
    # Group results by p_type (template id). Also keep an 'all' bucket.
    buckets = {}
    buckets['all'] = {'rows': [], 'errors_km': [], 'parsed_count': 0, 'total': 0, 'failures': []}

    for path in input_paths:
        with path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                p_type = str(obj.get('p_type', 'unknown'))
                if p_type not in buckets:
                    buckets[p_type] = {'rows': [], 'errors_km': [], 'parsed_count': 0, 'total': 0, 'failures': []}

                for key in [p_type, 'all']:
                    buckets[key]['total'] += 1

                true_lat = obj.get('Latitude')
                true_lon = obj.get('Longitude')
                idx = obj.get('loop_index')
                city = obj.get('AccentCity') or obj.get('City')
                pred_text = obj.get('output') or obj.get('full_output') or ''

                pred = extract_latlon(pred_text)
                if pred is None:
                    buckets[p_type]['rows'].append([idx, city, true_lat, true_lon, None, None, None, False, pred_text, str(path)])
                    buckets[p_type]['failures'].append(obj)
                    buckets['all']['rows'].append([idx, city, true_lat, true_lon, None, None, None, False, pred_text, str(path)])
                    buckets['all']['failures'].append(obj)
                    continue

                pred_lat, pred_lon = pred
                err = None
                if true_lat is not None and true_lon is not None:
                    try:
                        err = haversine_km(float(true_lat), float(true_lon), float(pred_lat), float(pred_lon))
                    except Exception:
                        err = None

                # Append to the specific p_type bucket
                buckets[p_type]['rows'].append([idx, city, true_lat, true_lon, pred_lat, pred_lon, err, True, pred_text, str(path)])
                if err is not None:
                    buckets[p_type]['errors_km'].append(err)
                else:
                    # keep consistency
                    pass
                if pred is not None:
                    buckets[p_type]['parsed_count'] += 1

                # Also append to 'all'
                buckets['all']['rows'].append([idx, city, true_lat, true_lon, pred_lat, pred_lon, err, True, pred_text, str(path)])
                if err is not None:
                    buckets['all']['errors_km'].append(err)
                buckets['all']['parsed_count'] += 1

    return buckets

def format_summary(model, shot, input_paths, analysis):
    total = analysis['total']
    parsed_count = analysis['parsed_count']
    errors_km = analysis['errors_km']

    summary = {
        'model': model,
        'shot': shot,
        'input_files': [str(p) for p in input_paths],
        'total_samples': total,
        'parsed_count': parsed_count,
        'parse_rate': parsed_count/total if total else 0.0
    }

    if errors_km:
        summary.update({
            'mean_error_km': mean(errors_km),
            'median_error_km': median(errors_km),
            'rmse_km': math.sqrt(mean([e*e for e in errors_km])),
        })
        for km in [1,5,10,50,100]:
            within = sum(1 for e in errors_km if e <= km)
            summary[f'within_{km}km'] = within
            summary[f'within_{km}km_rate'] = within/len(errors_km)

    return summary

def clean_previous_reports(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    patterns = ['eval_details_*', 'eval_summary_*', 'eval_failures_*', 'eval_report_*']
    for pat in patterns:
        for p in out_dir.glob(pat):
            try:
                p.unlink()
            except Exception:
                pass

def load_templates(templates_dir):
    templates = {}
    tdir = Path(templates_dir)
    for name in ['LLAMA2-CHAT-0.json','LLAMA2-CHAT-1.json']:
        p = tdir / name
        if p.exists():
            try:
                templates[name] = json.load(p.open('r', encoding='utf-8'))
            except Exception as e:
                templates[name] = f'<<ERROR LOADING: {e}>>'
        else:
            templates[name] = None
    return templates

# removed per-configuration text writer; aggregate report is used instead

def write_aggregate_report(out_dir, model, shot, buckets, templates, timestamp=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    report_path = out_dir / f'eval_report_{model}-{shot}-ALL-{ts}.txt'
    with report_path.open('w', encoding='utf-8') as rf:
        rf.write(f'Aggregated Evaluation Report - model={model} shot={shot}\n')
        rf.write(f'Timestamp: {ts}\n')
        rf.write('\n-- Per-Configuration Summaries --\n')
        for p_type, bucket in sorted(buckets.items()):
            total = bucket.get('total', 0)
            parsed = bucket.get('parsed_count', 0)
            errors = bucket.get('errors_km', [])
            rf.write(f'\n== p_type: {p_type} ==\n')
            rf.write(f'total_samples: {total}\n')
            rf.write(f'parsed_count: {parsed}\n')
            rf.write(f'parse_rate: {parsed/total if total else 0.0:.4f}\n')
            if errors:
                mf = mean(errors)
                md = median(errors)
                rmse = math.sqrt(mean([e*e for e in errors]))
                rf.write(f'mean_error_km: {mf:.4f}\n')
                rf.write(f'median_error_km: {md:.4f}\n')
                rf.write(f'rmse_km: {rmse:.4f}\n')
                for km in [1,5,10,50,100]:
                    within = sum(1 for e in errors if e <= km)
                    rf.write(f'within_{km}km: {within} ({within/len(errors):.3f})\n')
            else:
                rf.write('No parsed errors available.\n')

        rf.write('\n-- Templates Used --\n')
        for name, content in templates.items():
            rf.write(f'\n== {name} ==\n')
            if content is None:
                rf.write('MISSING\n')
            elif isinstance(content, str):
                rf.write(content + '\n')
            else:
                rf.write(json.dumps(content, ensure_ascii=False, indent=2) + '\n')

    return report_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='gpt-oss')
    p.add_argument('--shots', choices=['zero-shot','3-shot','both'], default='both')
    p.add_argument('--input', default=None)
    p.add_argument('--out_dir', default='outputs')
    p.add_argument('--templates_dir', default='templates')
    args = p.parse_args()

    shots_to_run = [args.shots] if args.shots != 'both' else ['zero-shot','3-shot']

    for shot in shots_to_run:
        if args.input:
            input_paths = [Path(args.input)]
        else:
            input_paths = sorted(Path(args.out_dir).glob(f'gen_coor_{args.model}-{shot}-*.jsonl'))

        if not input_paths:
            print(f'No input files found for model={args.model} shot={shot}. Skipping.')
            continue

        analysis_buckets = analyze_files(input_paths)
        templates = load_templates(args.templates_dir)

        # Produce one single aggregated TXT report containing all configurations
        report_path = write_aggregate_report(args.out_dir, args.model, shot, analysis_buckets, templates)
        print(f'Wrote aggregated report: {report_path}')

if __name__ == '__main__':
    main()
