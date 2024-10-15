import json, shutil

def fixed_number(n):
    n = str(n)
    while len(n) < 7:
        n = '0' + n
    return n

with open('choose.txt') as f:
    nums = json.load(f)
    destdir = 'tocopy'
    l = len(nums)
    count = 1
    for i in nums:
        n = str(fixed_number(i))
        name = 'model_{}.pth'.format(n)
        shutil.copyfile('output\\LP_detection\\' + name, destdir+'\\'+name)
        print(count, 'out of', l)
        count += 1