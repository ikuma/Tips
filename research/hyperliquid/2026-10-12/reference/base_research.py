#!/usr/bin/env python3
"""Exhaustive, causal reversal study. All reported prices are Hyperliquid.

Run fetch_data.py first. Outputs are UTF-8 CSV, with Dubai timestamps.
No interpolation of missing prices, OI, funding or taker volumes is used.
The finite grid is specified in config.json BEFORE examining outcomes.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, pathlib, time
import numpy as np
import pandas as pd
from scipy import stats
from fetch_data import COINS, ROOT, NORM

C=json.loads((ROOT/'config.json').read_text())
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
HOUR=pd.Timedelta(hours=1); FIVE=pd.Timedelta(minutes=5)
START=pd.Timestamp(C['research_start_utc']); END=pd.Timestamp(C['research_end_exclusive_utc'])
SPLIT=START+(END-START)/2
FRACTIONS=C['target_fractions']; HOLD=C['max_holding_hours']; SLOTS=C['evaluation_hours']+1
FAMILIES=175500 # (25*7*13*10 + 25*11*13*10) * 3 exits, including nulls
RATE=C['fee_bps_per_leg']/10000; SLIP=C['slippage_bps_per_leg']/10000

def dubai(t): return pd.Timestamp(t).tz_convert('Asia/Dubai').isoformat()

def read_candles(coin,tf='1h'):
    p=NORM/f'{coin}_hl_{tf}.csv.gz'
    if not p.exists():return pd.DataFrame()
    d=pd.read_csv(p); d.index=pd.to_datetime(d.t,unit='ms',utc=True)+pd.Timedelta(tf)
    d=d[~d.index.duplicated()].sort_index()
    for k in ['o','h','l','c','v']:d[k]=pd.to_numeric(d[k])
    return d[d.index<=END]

def wilder_atr(d,n=14):
    prev=d.c.shift(); tr=pd.concat([d.h-d.l,(d.h-prev).abs(),(d.l-prev).abs()],axis=1).max(axis=1).to_numpy()
    a=np.full(len(tr),np.nan)
    if len(tr)>=n:
        a[n-1]=np.mean(tr[:n])
        for k in range(n,len(tr)):a[k]=(a[k-1]*(n-1)+tr[k])/n
    return a

def asof_values(series,times,tolerance):
    return series.reindex(pd.DatetimeIndex(times),method='ffill',tolerance=tolerance).to_numpy(float)

def load(coin):
    d=read_candles(coin)
    if d.empty:return None
    d=d.reindex(pd.date_range(d.index.min(),d.index.max(),freq='h'))
    d['atr']=wilder_atr(d,C['atr_period_hours'])
    # A true lower bound on quote turnover, rather than volume*close pretending to be VWAP.
    d['vol24_lower']=(d.v*d.l).rolling(24,min_periods=24).sum()
    d['ret24']=d.c/d.c.shift(24)-1; d['ret168']=d.c/d.c.shift(168)-1
    fp=NORM/f'{coin}_hl_fundingHistory.csv.gz'; mp=NORM/f'{coin}_bn_metrics.csv.gz'; kp=NORM/f'{coin}_bn_5m.csv.gz'
    if not all(p.exists() for p in [fp,mp,kp]): return None
    f=pd.read_csv(fp); f.index=pd.to_datetime(f.time,unit='ms',utc=True)
    fr=f.fundingRate.astype(float).sort_index(); fr=fr[~fr.index.duplicated()]
    # Funding cash flows use the prior completed HL close as the disclosed oracle-price proxy.
    payments=fr.copy();payments.index=payments.index.floor('h')
    if payments.index.duplicated().any():raise ValueError('More than one funding settlement per hour')
    d['fund_payment_rate']=payments.reindex(d.index-HOUR).to_numpy()
    known=asof_values(fr,d.index-FIVE,HOUR)
    d['fund_known']=known*8
    for w in C['funding_hot_lookback_hours']:
        d[f'fmax{w}']=pd.Series(known*8,index=d.index).rolling(w,min_periods=w).max()
        d[f'fmin{w}']=pd.Series(known*8,index=d.index).rolling(w,min_periods=w).min()
    for w in C['funding_cool_mean_hours']:
        d[f'fmean{w}']=pd.Series(known*8,index=d.index).rolling(w,min_periods=w).mean()
    m=pd.read_csv(mp); m.index=pd.to_datetime(m.create_time,utc=True)
    oi=m.sum_open_interest.astype(float).sort_index();oi=oi[~oi.index.duplicated()]
    known_times=d.index-FIVE
    d['oi']=asof_values(oi,known_times,pd.Timedelta(minutes=C['max_oi_staleness_minutes']))
    for w in C['oi_window_hours']:
        prior=asof_values(oi,known_times-pd.Timedelta(hours=w),pd.Timedelta(minutes=C['max_oi_staleness_minutes']))
        d[f'oi{w}']=d.oi.to_numpy()/prior-1
    k=pd.read_csv(kp)
    k.index=pd.to_datetime(k.open_time,unit='ms',utc=True)+FIVE
    k=k[~k.index.duplicated()].sort_index()
    k=k.reindex(pd.date_range(k.index.min(),k.index.max(),freq='5min'))
    buy=k.taker_buy_volume.astype(float); sell=k.volume.astype(float)-buy
    bad=(buy<0)|(sell<0);buy[bad]=np.nan;sell[bad]=np.nan
    for w in C['taker_window_minutes']:
        n=w//5; ratio=buy.rolling(n,min_periods=n).sum()/sell.rolling(n,min_periods=n).sum()
        d[f'taker{w}']=ratio.reindex(d.index-FIVE).to_numpy()
    required=['atr','fund_known','fmax24','fmean3','oi1','oi3','oi6','oi12','taker15','taker60','taker240']
    d['features_ok']=np.isfinite(d[required]).all(axis=1)
    return dict(hour=d,five=read_candles(coin,'5m'),fund=fr,metrics=m,binance=k)

def detect_events(coin, data, btc):
    # Left-censored trends at the first usable observation are not fresh events.
    d=data['hour']; events=[]; armed=False; quiet=0; lock_until=-1
    for i in range(168,len(d)-SLOTS-HOLD+1):
        if d.index[i]<START or not np.isfinite(d.c.iloc[i]):continue
        r24,r168=d.ret24.iloc[i],d.ret168.iloc[i]
        up=r24>=C['event_24h_return'] or r168>=C['event_168h_return']
        dn=r24<=-C['event_24h_return'] or r168<=-C['event_168h_return']
        if not up and not dn:
            quiet+=1
            if quiet>=C['rearm_quiet_hours'] and i>=lock_until:armed=True
            continue
        quiet=0
        if up and dn:continue
        if not armed or i<lock_until:continue
        armed=False;lock_until=i+C['same_coin_event_lockout_hours']
        direction=1 if up else -1
        lookback=24 if direction*r24>=C['event_24h_return'] else 168
        if d.vol24_lower.iloc[i]<C['volume_lower_bound_usd'] or not d.features_ok.iloc[i]:continue
        segment=d.iloc[i-lookback+1:i+SLOTS+HOLD]
        if segment[['o','h','l','c']].isna().any().any():continue
        if segment.index[-1]>END:continue
        ts=d.index[i]; b=btc.reindex([ts]).iloc[0]
        regime='up' if b>=C['btc_regime_7d_threshold'] else 'down' if b<=-C['btc_regime_7d_threshold'] else 'sideways'
        events.append(dict(event_id=f'{coin}_{int(ts.timestamp())}',coin=coin,i=i,t0=ts,
           direction=direction,lookback=lookback,anchor=float(d.c.iloc[i-lookback]),
           ret24=float(r24),ret168=float(r168),volume_lower_usd=float(d.vol24_lower.iloc[i]),
           regime=regime,btc_return_7d=float(b)))
    return events

def parameter_grid(include_five=False):
    f=[None]+[dict(hot=h,cool=c,hot_h=w,cool_h=a) for h,c,w,a in itertools.product(
       C['funding_hot_8h_rate'],C['funding_cool_8h_rate'],C['funding_hot_lookback_hours'],C['funding_cool_mean_hours'])]
    p=[None]+[dict(tf='1h',bars=n) for n in C['price_1h_no_extreme_bars']]
    if include_five:p += [dict(tf='5m',bars=n) for n in C['price_5m_no_extreme_bars']]
    o=[None]+[dict(hours=w,drop=a) for w,a in itertools.product(C['oi_window_hours'],C['oi_drop_fraction'])]
    t=[None]+[dict(minutes=w,ratio=a) for w,a in itertools.product(C['taker_window_minutes'],C['directional_taker_ratio_max'])]
    return [f,p,o,t]

def simulate_path(d, decision_i, direction, extreme, anchor, fraction, slip=SLIP):
    """Return net R, exit offset, stop/target, and hourly marked-to-market increments."""
    a=d.attrs.get('sim_arrays')
    if a is None:a={k:d[k].to_numpy(float) for k in ['o','h','l','c','atr','fund_payment_rate']}
    q=-direction; entry_i=decision_i+1
    entry=float(a['o'][entry_i])*(1+q*slip)
    atr=float(a['atr'][decision_i]); target=extreme+fraction*(anchor-extreme)
    if direction==1: stop=max(extreme*(1+C['extreme_stop_buffer']),entry+C['atr_multiple']*atr)
    else:stop=min(extreme*(1-C['extreme_stop_buffer']),entry-C['atr_multiple']*atr)
    risk=abs(stop-entry)
    if not np.isfinite(risk) or risk<=0 or stop<=0 or q*(target-entry)<=0:
        return None
    stop_hit=(a['h'][entry_i:entry_i+HOLD]>=stop) if q<0 else (a['l'][entry_i:entry_i+HOLD]<=stop)
    target_hit=(a['l'][entry_i:entry_i+HOLD]<=target) if q<0 else (a['h'][entry_i:entry_i+HOLD]>=target)
    hits=np.flatnonzero(stop_hit|target_hit);ex=int(hits[0]) if len(hits) else HOLD-1
    if stop_hit[ex]:
        px=max(a['o'][entry_i+ex],stop) if q<0 else min(a['o'][entry_i+ex],stop);reason='stop'
    elif target_hit[ex]:px=target;reason='target'
    else:px=a['c'][entry_i+ex];reason='timeout'
    rates=a['fund_payment_rate'][entry_i:entry_i+HOLD].copy();rates[0]=0
    if not np.isfinite(rates[:ex+1]).all():return None
    cash=-q*a['c'][entry_i-1:entry_i+HOLD-1]*rates/risk
    accumulated=np.cumsum(cash);funding=float(accumulated[ex])
    vals=q*(a['c'][entry_i:entry_i+HOLD]-entry)/risk-RATE*entry/risk+accumulated
    px*=1-q*slip
    vals[ex]=q*(px-entry)/risk-RATE*(entry+px)/risk+funding
    vals[ex+1:]=vals[ex]
    changes=np.diff(np.r_[0.,vals])
    return dict(r=float(vals[-1]),exit_offset=ex,stop=stop,target=target,entry=entry,risk=risk,
                exit_price=float(px),fee_R=RATE*(entry+px)/risk,price_R=q*(px-entry)/risk,
                funding_r=funding,reason=reason,path=changes)

def build(events,data,include_five=False):
    n=len(events); parameters=parameter_grid(include_five)
    masks=[np.ones((len(x),n,SLOTS),dtype=bool) for x in parameters]
    returns=np.full((n,SLOTS,3),np.nan); exits=np.full((n,SLOTS,3),-1,dtype=int)
    paths=np.zeros((n,SLOTS,3,HOLD),dtype=np.float32)
    details={}; entry_global=np.zeros((n,SLOTS),int)
    global_start=min(x['hour'].index.min() for x in data.values())
    valid=np.zeros((n,SLOTS),dtype=bool)
    for e,ev in enumerate(events):
        d=data[ev['coin']]['hour'];i=ev['i'];direction=ev['direction']
        if 'sim_arrays' not in d.attrs:d.attrs['sim_arrays']={k:d[k].to_numpy(float) for k in ['o','h','l','c','atr','fund_payment_rate']}
        field='h' if direction==1 else 'l'
        initial=d[field].iloc[i-ev['lookback']+1:i+1].to_numpy()
        extreme=np.max(initial) if direction==1 else np.min(initial)
        last=i-ev['lookback']+1+np.flatnonzero(initial==extreme)[-1]
        five=data[ev['coin']]['five']
        for j in range(SLOTS):
            z=i+j;row=d.iloc[z];ts=d.index[z]
            new=row[field]
            if direction*(new-extreme)>=0:extreme=float(new);last=z
            entry_global[e,j]=int((ts-global_start)/HOUR)+1
            valid[e,j]=bool(row.features_ok)
            for k,param in enumerate(parameters[0][1:],1):
                hot=row[f'fmax{param["hot_h"]}'] if direction==1 else -row[f'fmin{param["hot_h"]}']
                cool=direction*row[f'fmean{param["cool_h"]}']
                masks[0][k,e,j]=hot>=param['hot'] and cool<=param['cool']
            for k,param in enumerate(parameters[1][1:],1):
                if param['tf']=='1h': ok=z-last>=param['bars']
                elif five.empty:ok=False
                else:
                    part=five.loc[(five.index>ts-param['bars']*FIVE)&(five.index<=ts),field]
                    ok=len(part)==param['bars'] and (part.max()<extreme if direction==1 else part.min()>extreme)
                masks[1][k,e,j]=ok
            for k,param in enumerate(parameters[2][1:],1):masks[2][k,e,j]=row[f'oi{param["hours"]}']<=-param['drop']
            for k,param in enumerate(parameters[3][1:],1):
                ratio=row[f'taker{param["minutes"]}'];ratio=ratio if direction==1 else 1/ratio
                masks[3][k,e,j]=np.isfinite(ratio) and ratio<=param['ratio']
            for target,fraction in enumerate(FRACTIONS):
                result=simulate_path(d,z,direction,extreme,ev['anchor'],fraction)
                if result is None:continue
                returns[e,j,target]=result['r'];exits[e,j,target]=entry_global[e,j]+result['exit_offset']
                paths[e,j,target]=result.pop('path');details[(e,j,target)]=result
        if e%25==0:print('EVENT CACHE',e,'/',n,flush=True)
    packed=[]
    weights=np.left_shift(np.uint64(1),np.arange(64,dtype=np.uint64))
    for a in masks:
        a=a&valid[None,:,:]
        lo=(a[:,:,:64]*weights).sum(axis=2,dtype=np.uint64)
        hi=(a[:,:,64:]*weights[:SLOTS-64]).sum(axis=2,dtype=np.uint64)
        packed.append((lo,hi))
    return dict(events=events,parameters=parameters,packed=packed,r=returns,exits=exits,paths=paths,
                detail=details,entry_global=entry_global,global_start=global_start)

def first_slots(cache,ids):
    lo=np.full(len(cache['events']),np.iinfo(np.uint64).max,dtype=np.uint64)
    hi=lo.copy()
    for family,k in enumerate(ids):lo &= cache['packed'][family][0][k];hi &= cache['packed'][family][1][k]
    def bit(x):
        low=x & (~x+np.uint64(1));return np.log2(np.maximum(low,np.uint64(1)).astype(float)).astype(int)
    return np.where(lo!=0,bit(lo),np.where(hi!=0,64+bit(hi),-1))

def values(cache,slots,target,mask):
    indices=np.flatnonzero(mask&(slots>=0)); j=slots[indices]
    r=cache['r'][indices,j,target];ok=np.isfinite(r)
    return indices[ok],j[ok],r[ok]

def cluster_stats(x, groups):
    n=len(x)
    if n==0:return dict(n=0,mean=np.nan,se=np.nan,lo=np.nan,hi=np.nan,p=np.nan,clusters=0)
    mean=float(np.mean(x));_,idx=np.unique(groups,return_inverse=True);b=idx.max()+1
    if n<2 or b<2:return dict(n=n,mean=mean,se=np.nan,lo=np.nan,hi=np.nan,p=np.nan,clusters=b)
    sums=np.bincount(idx,weights=x-mean)
    se=math.sqrt(b/(b-1)*np.sum(sums*sums))/n
    critical=stats.t.ppf(.975,b-1)
    p=float(stats.t.sf(mean/se,b-1)) if se>0 else (0. if mean>0 else 1.)
    return dict(n=n,mean=mean,se=se,lo=mean-critical*se,hi=mean+critical*se,p=p,clusters=b)

def mark_to_market_dd(cache,indices,slots,target):
    if len(indices)==0:return np.nan
    positions=cache['entry_global'][indices,slots,None]+np.arange(HOLD)[None,:]
    increments=cache['paths'][indices,slots,target,:]
    curve=np.cumsum(np.bincount(positions.ravel(),weights=increments.ravel()))
    return float(np.max(np.maximum.accumulate(np.r_[0.,curve])[1:]-curve))

def score(cache,slots,target,mask,dd=True):
    indices,j,r=values(cache,slots,target,mask)
    ev=cache['events']; groups=np.array([int(ev[x]['t0'].timestamp())//(7*86400) for x in indices])
    s=cluster_stats(r,groups)
    all_i=np.flatnonzero(mask&np.isfinite(cache['r'][:,0,target]))
    base=cache['r'][all_i,0,target]
    strategy=np.zeros(len(all_i));pos={v:k for k,v in enumerate(all_i)}
    for ii,rr in zip(indices,r):
        if ii in pos:strategy[pos[ii]]=rr
    delta=strategy-base
    dg=np.array([int(ev[x]['t0'].timestamp())//(7*86400) for x in all_i])
    ds=cluster_stats(delta,dg)
    wins=r[r>0];loss=r[r<=0]
    ans=dict(n=s['n'],win_rate=float(np.mean(r>0)) if len(r) else np.nan,
       avg_winner_R=float(np.mean(wins)) if len(wins) else np.nan,
       avg_loser_R=float(np.mean(loss)) if len(loss) else np.nan,
       mean_net_R=s['mean'],expectancy_R=s['mean'],ci95_low_R=s['lo'],ci95_high_R=s['hi'],
       se_R=s['se'],week_clusters=s['clusters'],p_positive=s['p'],
       paired_n=ds['n'],per_event_delta_R=ds['mean'],delta_ci95_low=ds['lo'],delta_ci95_high=ds['hi'],
       p_improve_baseline=ds['p'],p_bonf_positive=min(1,s['p']*FAMILIES) if np.isfinite(s['p']) else np.nan,
       p_bonf_improve=min(1,ds['p']*FAMILIES) if np.isfinite(ds['p']) else np.nan,
       matched_baseline_R=float(np.nanmean(cache['r'][indices,0,target])) if len(indices) and np.isfinite(cache['r'][indices,0,target]).any() else np.nan,
       matched_baseline_n=int(np.isfinite(cache['r'][indices,0,target]).sum()),
       max_drawdown_mtm_R=mark_to_market_dd(cache,indices,j,target) if dd else np.nan)
    return ans

def rule_record(cache,ids,target):
    desc=[cache['parameters'][i][x] for i,x in enumerate(ids)]
    return dict(rule_id='-'.join(map(str,ids)),target_fraction=FRACTIONS[target],conditions=sum(x>0 for x in ids),
       funding=json.dumps(desc[0],separators=(',',':')),price=json.dumps(desc[1],separators=(',',':')),
       oi=json.dumps(desc[2],separators=(',',':')),taker=json.dumps(desc[3],separators=(',',':')))

def split_masks(events):
    # Deterministic coin holdout fixed without looking at any PnL or outcomes.
    held=set(sorted(COINS,key=lambda c:hashlib.sha256(('20261012:'+c).encode()).hexdigest())[:5])
    train=np.array([e['t0']+pd.Timedelta(hours=144)<SPLIT and e['coin'] not in held for e in events])
    test=np.array([e['t0']>=SPLIT and e['coin'] not in held for e in events])
    cross=np.array([e['t0']>=SPLIT and e['coin'] in held for e in events])
    coin_train_time=np.array([e['t0']+pd.Timedelta(hours=144)<SPLIT and e['coin'] in held for e in events])
    return held,dict(train=train,test=test,unseen_coin_test=cross,unseen_coin_first_half=coin_train_time,all=np.ones(len(events),bool))

def search(cache,cohort='hourly_6m'):
    import csv
    events=cache['events'];held,masks=split_masks(events)
    if cohort!='hourly_6m':
        # A short supplementary cohort cannot claim six-month out-of-sample validation.
        masks['train']=np.ones(len(events),bool)
    baseline=[];null=np.zeros(len(events),int)
    for name,mask in masks.items():
        for k in range(3):baseline.append(dict(cohort=cohort,split=name,**rule_record(cache,(0,0,0,0),k),**score(cache,null,k,mask)))
    pd.DataFrame(baseline).to_csv(OUT/f'{cohort}_baseline.csv',index=False)
    picks=[];best=[];nrules=math.prod(len(x) for x in cache['parameters'])
    path=OUT/f'{cohort}_all_grid_train.csv'
    with path.open('w',newline='') as f:
        writer=None
        for number,ids in enumerate(itertools.product(*[range(len(x)) for x in cache['parameters']])):
            slots=first_slots(cache,ids)
            for target in range(3):
                result=score(cache,slots,target,masks['train'])
                row=dict(cohort=cohort,split='train' if cohort=='hourly_6m' else 'short_sample_exploratory',**rule_record(cache,ids,target),**result)
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                writer.writerow(row)
                qualifies=(sum(ids)>0 and result['n']>=C['minimum_train_trades'] and
                    result['week_clusters']>=C['minimum_active_week_clusters'] and result['p_bonf_positive']<.05 and result['p_bonf_improve']<.05)
                if qualifies:picks.append((result['per_event_delta_R'],ids,target))
                if sum(ids)>0 and result['n']>=C['minimum_diagnostic_train_trades']:
                    best.append((result['ci95_low_R'],ids,target))
            if number%1000==0:print('GRID',cohort,number,'/',nrules,flush=True)
    qualifying_train_count=len(picks)
    picks=sorted(picks,reverse=True)[:3]
    # Freeze at most 3 diagnostic candidates on training data even if none passes correction.
    diagnostics=[];seen=set()
    for _,ids,target in sorted(best,reverse=True):
        if ids in seen:continue
        diagnostics.append((ids,target));seen.add(ids)
        if len(diagnostics)==3:break
    selected=[(ids,t) for _,ids,t in picks]
    eval_rules=list(dict.fromkeys(selected+diagnostics+[((0,0,0,0),k) for k in range(3)]))
    out=[];trades=[];recommend=[]
    for ids,target in eval_rules:
        slots=null if ids==(0,0,0,0) else first_slots(cache,ids)
        for name,mask in masks.items():
            s=score(cache,slots,target,mask);out.append(dict(cohort=cohort,split=name,selected_after_bonf=(ids,target) in selected,
                diagnostic_only=(ids,target) not in selected and ids!=(0,0,0,0),**rule_record(cache,ids,target),**s))
            if name in ['test','unseen_coin_test'] and (ids,target) in selected and s['n']>=C['minimum_test_trades'] and s['ci95_low_R']>0 and s['delta_ci95_low']>0:
                recommend.append(dict(rule_id='-'.join(map(str,ids)),target_fraction=FRACTIONS[target],split=name))
        for rg in ['up','down','sideways']:
            for period in ['train','test','unseen_coin_test','all']:
                m=masks[period]&np.array([e['regime']==rg for e in events])
                out.append(dict(cohort=cohort,split=period+'__BTC_'+rg,**rule_record(cache,ids,target),**score(cache,slots,target,m)))
        for side in [1,-1]:
            m=masks['test']&np.array([e['direction']==side for e in events])
            out.append(dict(cohort=cohort,split='test__'+('short_after_rally' if side==1 else 'long_after_drop'),**rule_record(cache,ids,target),**score(cache,slots,target,m)))
        for coin in COINS:
            m=masks['all']&np.array([e['coin']==coin for e in events])
            out.append(dict(cohort=cohort,split='coin__'+coin,**rule_record(cache,ids,target),**score(cache,slots,target,m)))
        indices,j,rs=values(cache,slots,target,masks['all'])
        for e,slot,r in zip(indices,j,rs):
            ev=events[e];detail=cache['detail'][(e,slot,target)]
            trades.append(dict(cohort=cohort,rule_id='-'.join(map(str,ids)),target_fraction=FRACTIONS[target],event_id=ev['event_id'],coin=ev['coin'],
               event_dubai=dubai(ev['t0']),entry_dubai=dubai(ev['t0']+slot*HOUR),
               exit_bar_end_dubai=dubai(ev['t0']+(slot+detail['exit_offset']+1)*HOUR),
               direction='short' if ev['direction']==1 else 'long',regime=ev['regime'],
               coin_holdout=ev['coin'] in held,**detail))
    pd.DataFrame(out).to_csv(OUT/f'{cohort}_validation.csv',index=False)
    pd.DataFrame(trades).to_csv(OUT/f'{cohort}_trades.csv',index=False)
    (OUT/f'{cohort}_selection.json').write_text(json.dumps(dict(selected=selected,diagnostic=diagnostics,
       qualifying_train_count=qualifying_train_count,validation_passes=recommend,coin_holdouts=sorted(held)),indent=2))
    return dict(baseline=baseline,selected=selected,diagnostic=diagnostics)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['hourly','five','both'],default='both');a=ap.parse_args()
    data={};coverage=[]
    for coin in COINS:
        x=load(coin)
        if x is None:print('UNAVAILABLE',coin,flush=True);continue
        if x['hour'].index.min()>START-pd.Timedelta(days=7)+HOUR or x['hour'].index.max()<END:
            print('INSUFFICIENT HL HISTORY',coin,flush=True);continue
        data[coin]=x
        for name,key in [('hl_1h','hour'),('hl_5m','five'),('hl_funding','fund'),('binance_oi','metrics'),('binance_taker_5m','binance')]:
            d=x[key]
            coverage.append(dict(coin=coin,dataset=name,rows=len(d),start_dubai=dubai(d.index.min()) if len(d) else None,
               end_dubai=dubai(d.index.max()) if len(d) else None,
               observed_candles=int(d.c.notna().sum()) if key in ['hour','five'] else None,
               complete_feature_hours=int(d.features_ok.sum()) if key=='hour' else None))
        print('LOADED',coin,len(x['hour']),flush=True)
    pd.DataFrame(coverage).to_csv(OUT/'data_coverage.csv',index=False)
    if 'BTC' not in data:raise RuntimeError('BTC missing: regimes cannot be assigned')
    btc=data['BTC']['hour'].ret168
    events=sum([detect_events(c,d,btc) for c,d in data.items()],[])
    events=sorted(events,key=lambda e:(e['t0'],e['coin']))
    ed=pd.DataFrame(events);ed['t0']=ed['t0'].map(dubai);ed.to_csv(OUT/'events.csv',index=False)
    held,_=split_masks(events)
    pd.DataFrame([dict(coin=c,fold='unseen' if c in held else 'development') for c in COINS]).to_csv(OUT/'coin_folds.csv',index=False)
    print('EVENTS',len(events),'split',dubai(SPLIT),'holdout',sorted(held),flush=True)
    if a.cohort in ['hourly','both']:
        cache=build(events,data);search(cache)
        del cache
    if a.cohort in ['five','both']:
        first5=max(d['five'].index.min() for d in data.values() if not d['five'].empty)+pd.Timedelta(hours=2)
        ev5=[e for e in events if not data[e['coin']]['five'].empty and e['t0']>=first5]
        print('FIVE MINUTE EVENTS',len(ev5),flush=True)
        if ev5:
            cache=build(ev5,data,True);search(cache,'five_min_short')
    print('DONE',flush=True)

if __name__=='__main__':main()
