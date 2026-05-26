import csv
from collections import defaultdict

# 读取含时间嵌入的实验数据
rows = list(csv.DictReader(open(r'F:\paper\code\csv\uniform_time_5cv_temporal_embedding.csv')))
groups = defaultdict(lambda: defaultdict(list))

for r in rows:
    for m in ['r2','rmse','mae']:
        groups[(r['feature_set'], r['model'])][m].append(float(r[m]))

print('='*80)
print('含时间嵌入的各特征集 - 5折交叉验证结果')
print('='*80)
print(f'{"Feature_Set":35} {"Model":12} {"R2_avg":10} {"R2_std":8} {"RMSE_avg":10} {"RMSE_std":8} {"MAE_avg":10} {"MAE_std":8}')
print('-'*80)

for k, v in sorted(groups.items()):
    r2s = v['r2']
    rmses = v['rmse']
    maes = v['mae']
    r2_avg = sum(r2s)/len(r2s)
    r2_std = (sum((x-r2_avg)**2 for x in r2s)/len(r2s))**0.5
    rmse_avg = sum(rmses)/len(rmses)
    rmse_std = (sum((x-rmse_avg)**2 for x in rmses)/len(rmses))**0.5
    mae_avg = sum(maes)/len(maes)
    mae_std = (sum((x-mae_avg)**2 for x in maes)/len(maes))**0.5
    print(f'{k[0]:35} {k[1]:12} {r2_avg:.4f}   {r2_std:.4f}  {rmse_avg:.4f}   {rmse_std:.4f}  {mae_avg:.4f}   {mae_std:.4f}')

# 读取静态基线结果
print()
print('='*70)
print('静态基线 (uniform_time_5cv_results.csv)')
print('='*70)
rows2 = list(csv.DictReader(open(r'F:\paper\code\csv\uniform_time_5cv_results.csv')))
groups2 = defaultdict(lambda: defaultdict(list))
for r in rows2:
    for m in ['r2','rmse','mae']:
        groups2[(r['dataset'], r['model'])][m].append(float(r[m]))
print(f'{"Dataset":25} {"Model":12} {"R2_avg":10} {"R2_std":8} {"RMSE_avg":10} {"RMSE_std":8}')
print('-'*70)
for k, v in sorted(groups2.items()):
    r2s = v['r2']
    rmses = v['rmse']
    r2_avg = sum(r2s)/len(r2s)
    r2_std = (sum((x-r2_avg)**2 for x in r2s)/len(r2s))**0.5
    rmse_avg = sum(rmses)/len(rmses)
    rmse_std = (sum((x-rmse_avg)**2 for x in rmses)/len(rmses))**0.5
    print(f'{k[0]:25} {k[1]:12} {r2_avg:.4f}   {r2_std:.4f}  {rmse_avg:.4f}   {rmse_std:.4f}')