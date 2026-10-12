#!/usr/bin/env python3
"""Fetch public data, cache immutable responses, and record every failure.

Only public/read-only endpoints are called. No trading credentials are needed.
Binance REST 451 is not bypassed; its separate public archive is used.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, io, json, pathlib, threading
import time, urllib.error, urllib.request, zipfile
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
NORM = ROOT / 'data' / 'normalized'
COINS = ['BTC','ETH','NEAR','HYPE','ZEC','SOL','STRK','PUMP','ENA','ONDO',
         'ZRO','XRP','W','WLD','VVV','GRASS','TAO','SUI','UNI','AVAX']
END = pd.Timestamp('2026-10-12T00:00:00Z')
START = pd.Timestamp('2026-03-18T00:00:00Z')
lock = threading.Lock()
records = []

def csv_atomic(d,path):
    temp=path.with_name(path.name+'.part')
    d.to_csv(temp,index=False,compression={'method':'gzip','mtime':0})
    # Finish and validate the compressed stream before installing the file.
    import gzip
    with gzip.open(temp,'rb') as f:
        while f.read(1024*1024):pass
    temp.replace(path)

def request(url, payload=None, attempts=3):
    for k in range(attempts):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload else None,
                                         headers={'Content-Type':'application/json','User-Agent':'QuantResearch/1.0'})
            with urllib.request.urlopen(req, timeout=25) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403,404,451): raise
            if k == attempts-1: raise
            time.sleep(2**k)
        except Exception:
            if k == attempts-1: raise
            time.sleep(2**k)

def log(source, path, status, nbytes=0, error='', digest=''):
    with lock:
        records.append(dict(source=source, file=str(path.relative_to(ROOT)), status=status,
                            bytes=nbytes, error=error, sha256=digest))

def hl(coin, kind, interval=None):
    path=RAW/'hl'/f'{coin}_{interval or kind}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        if path.exists(): data=json.loads(path.read_text())
        elif kind=='candleSnapshot':
            data=json.loads(request('https://api.hyperliquid.xyz/info',
                 {'type':kind,'req':{'coin':coin,'interval':interval,
                    'startTime':int(START.timestamp()*1000),'endTime':int(END.timestamp()*1000)-1}}))
            path.write_text(json.dumps(data))
        else:
            data=[]; cursor=int(START.timestamp()*1000)
            while cursor<int(END.timestamp()*1000):
                batch=json.loads(request('https://api.hyperliquid.xyz/info',
                     {'type':'fundingHistory','coin':coin,'startTime':cursor,
                      'endTime':int(END.timestamp()*1000)-1}))
                if not isinstance(batch,list): raise ValueError(str(batch))
                if not batch:break
                data.extend(batch); new=max(int(x['time']) for x in batch)+1
                if new<=cursor:raise ValueError('Non-advancing pagination')
                cursor=new
                time.sleep(.15)
            path.write_text(json.dumps(data))
        if not isinstance(data,list):raise ValueError(str(data))
        log('HL '+kind,path,'ok',path.stat().st_size,digest=hashlib.sha256(path.read_bytes()).hexdigest())
        return coin, interval or kind, len(data)
    except Exception as e:
        log('HL '+kind,path,'failed',error=str(e));return coin,interval or kind,str(e)

def archive(coin, typ, date, freq='daily'):
    symbol=coin+'USDT'; middle='5m/' if typ=='klines' else ''
    fname=f'{symbol}-'+('5m' if typ=='klines' else 'metrics')+f'-{date}.zip'
    url=f'https://data.binance.vision/data/futures/um/{freq}/{typ}/{symbol}/{middle}{fname}'
    path=RAW/'binance'/coin/fname
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        if not path.exists():
            b=request(url)
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                if z.testzip() is not None:raise ValueError('Corrupt ZIP')
            path.write_bytes(b)
        log(url,path,'ok',path.stat().st_size,digest=hashlib.sha256(path.read_bytes()).hexdigest())
        return True
    except Exception as e:
        log(url,path,'failed',error=str(e));return False

def normalize(coin):
    NORM.mkdir(parents=True,exist_ok=True)
    for kind in ['1h','5m','fundingHistory']:
        p=RAW/'hl'/f'{coin}_{kind}.json'
        if not p.exists():continue
        d=pd.DataFrame(json.loads(p.read_text()))
        if d.empty:continue
        d=d.drop_duplicates('time' if kind=='fundingHistory' else 't').sort_values('time' if kind=='fundingHistory' else 't')
        csv_atomic(d,NORM/f'{coin}_hl_{kind}.csv.gz')
    for typ in ['metrics','5m']:
        frames=[]
        for p in sorted((RAW/'binance'/coin).glob(f'*-{typ}-*.zip')):
            try:
                with zipfile.ZipFile(p) as z:d=pd.read_csv(z.open(z.namelist()[0]))
                frames.append(d)
            except Exception as e:print('PARSE ERROR',p,e,flush=True)
        if frames:
            d=pd.concat(frames,ignore_index=True)
            key='create_time' if typ=='metrics' else 'open_time'
            d=d.drop_duplicates(key).sort_values(key)
            columns=['create_time','symbol','sum_open_interest','sum_open_interest_value','sum_taker_long_short_vol_ratio'] if typ=='metrics' else ['open_time','close_time','volume','taker_buy_volume']
            d=d[columns]
            csv_atomic(d,NORM/f'{coin}_bn_{typ}.csv.gz')
            print('NORMALIZED',coin,typ,len(d),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','hl','binance','normalize','compact'],default='all')
    parser.add_argument('--workers',type=int,default=10);args=parser.parse_args()
    if args.stage in ['all','hl']:
        jobs=[(c,k,i) for c in COINS for k,i in [('candleSnapshot','1h'),('candleSnapshot','5m'),('fundingHistory',None)]]
        with cf.ThreadPoolExecutor(max_workers=3) as pool:
            for result in pool.map(lambda j:hl(*j),jobs):print('HL',result,flush=True)
    if args.stage in ['all','binance']:
        dates=pd.date_range(START,END-pd.Timedelta(days=1),freq='D')
        jobs=[(c,'metrics',d.strftime('%Y-%m-%d'),'daily') for c in COINS for d in dates]
        months=pd.date_range(START.replace(day=1),END.replace(day=1),freq='MS')
        jobs += [(c,'klines',m.strftime('%Y-%m'),'monthly') for c in COINS for m in months if m.month!=END.month]
        jobs += [(c,'klines',d.strftime('%Y-%m-%d'),'daily') for c in COINS for d in pd.date_range(END.replace(day=1),END-pd.Timedelta(days=1),freq='D')]
        ok=0
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i,v in enumerate(pool.map(lambda j:archive(*j),jobs),1):
                ok+=v
                if i%100==0:print('ARCHIVE',i,'/',len(jobs),'ok',ok,flush=True)
    if records:
        pd.DataFrame(records).to_csv(ROOT/f'download_manifest_{args.stage}.csv',index=False)
    if args.stage in ['all','normalize']:
        for c in COINS:normalize(c)
    if args.stage=='compact':
        for c in COINS:
            for typ in ['metrics','5m']:
                p=NORM/f'{c}_bn_{typ}.csv.gz'
                if not p.exists():continue
                columns=['create_time','symbol','sum_open_interest','sum_open_interest_value','sum_taker_long_short_vol_ratio'] if typ=='metrics' else ['open_time','close_time','volume','taker_buy_volume']
                d=pd.read_csv(p,usecols=columns);csv_atomic(d[columns],p)
            print('COMPACT',c,flush=True)

if __name__=='__main__':main()
