import json
perf = []
with open('E:\\Programowanie\\Camera-main\\metrics.json') as f:
    for d in json.load(f):
        perf.append((d['iteration'], d['total_loss']))
chosen = [i for i, l in perf if i >= 7000 and i< 8000]
perf.sort(key=lambda d: d[1])
add = [i for idx, (i, l) in enumerate(perf) if i not in chosen and idx <= 10]
chosen += add
with open('E:\\Programowanie\\Camera-main\\choose.txt', 'w') as f:
    json.dump(chosen[1:], f)
    
with open('choose.txt') as f:
    x = json.load(f)
    print((len(x))*0.282)