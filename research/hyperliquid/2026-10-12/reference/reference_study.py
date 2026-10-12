#!/usr/bin/env python3
"""Reference-indicator audit. No future peak is used as an executable exit.

Outputs distinguish favorable excursion, target-before-stop, and net PnL.
Repeated hourly landmarks are clustered by episode/week, not called trades.
"""
from __future__ import annotations
import json, math, pathlib, warnings
import numpy as np
import pandas as pd
from scipy import stats
import base_research as b
from fetch_data import ROOT,COINS

C=json.loads((ROOT/'reference_config.json').read_text())
OUT=ROOT/'reference_results';OUT.mkdir(exist_ok=True)
SIGNALS=['price_pause_1h','oi_drop_3pct_3h','price_plus_oi','funding_cool','taker_reversal_1h','price_oi_taker']
TARGETS=[('net1bp','bps',1),('net10bp','bps',10),('net25bp','bps',25),('net50bp','bps',50),
         ('R025','R',.25),('R05','R',.5),('R1','R',1.)]
METRICS=['further_extreme','mae_atr','mfe_net_atr','hit_net10bp','net10bp_R','hit_R05']

def scan_events(coin,data,btc):
    """Preserve the original causal reset; remove its 144h forward-data censor."""
    d=data['hour'];events=[];armed=False;quiet=0;lock=-1
    for i in range(168,len(d)-1):
        if d.index[i]<b.START or not np.isfinite(d.c.iloc[i]):continue
        r24,r168=d.ret24.iloc[i],d.ret168.iloc[i]
        up=r24>=.08 or r168>=.15;down=r24<=-.08 or r168<=-.15
        if not up and not down:
            quiet+=1
            if quiet>=24 and i>=lock:armed=True
            continue
        quiet=0
        if up and down:continue
        if not armed or i<lock:continue
        armed=False;lock=i+144;direction=1 if up else -1
        lookback=24 if direction*r24>=.08 else 168
        if not np.isfinite(d.vol24_lower.iloc[i]) or d.vol24_lower.iloc[i]<20e6 or not d.features_ok.iloc[i]:continue
        if d[['o','h','l','c']].iloc[i-lookback+1:i+1].isna().any().any():continue
        ts=d.index[i];br=btc.loc[ts]
        regime='up' if br>=.03 else 'down' if br<=-.03 else 'sideways'
        events.append(dict(event_id=f'{coin}_{int(ts.timestamp())}',coin=coin,i=i,t0=ts,direction=direction,
            lookback=lookback,anchor=float(d.c.iloc[i-lookback]),ret24=float(r24),ret168=float(r168),
            regime=regime,volume_lower_usd=float(d.vol24_lower.iloc[i]),btc_return_7d=float(br),
            full_window=ts+pd.Timedelta(hours=84)<=b.END))
    return events

def load_all():
    data={};coverage=[]
    for coin in COINS:
        with np.errstate(divide='ignore',invalid='ignore'):x=b.load(coin)
        if x is None:continue
        d=x['hour'];x['frames']={60:d,5:x['five']}
        p=ROOT/'data/raw_1m'/f'{coin}.json'
        if p.exists():
            minute=pd.DataFrame(json.loads(p.read_text()))
            if not minute.empty:
                minute.index=pd.to_datetime(minute.t,unit='ms',utc=True)+pd.Timedelta(minutes=1)
                minute=minute.sort_index();minute=minute[~minute.index.duplicated()]
                for k in ['o','h','l','c','v']:minute[k]=minute[k].astype(float)
                x['frames'][1]=minute[minute.index<=b.END]
        payments=x['fund'].copy();payments.index=payments.index.floor('h')
        prices=d.c.reindex(payments.index)
        # First payment preceding available HL prices is irrelevant to evaluated entries.
        cash=(payments*prices).fillna(0)
        x['fund_ns']=cash.index.asi8;x['fund_cum']=cash.cumsum().to_numpy(float)
        x['arrays']={}
        for res,frame in x['frames'].items():
            x['arrays'][res]={k:frame[k].to_numpy(float) for k in ['o','h','l','c']}
            x['arrays'][res]['time_ns']=frame.index.asi8
            coverage.append(dict(coin=coin,resolution_minutes=res,rows=len(frame),
                start_dubai=b.dubai(frame.index.min()) if len(frame) else None,
                end_dubai=b.dubai(frame.index.max()) if len(frame) else None))
        data[coin]=x
        print('LOADED',coin,flush=True)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    return data

def make_landmarks(events,data):
    held,_=b.split_masks(events);btc=data['BTC']['hour'];rows=[]
    for ev in events:
        d=data[ev['coin']]['hour'];i=ev['i'];direction=ev['direction'];field='h' if direction==1 else 'l'
        init=d[field].iloc[i-ev['lookback']+1:i+1].to_numpy()
        extreme=float(init.max() if direction==1 else init.min())
        last=i-ev['lookback']+1+np.flatnonzero(init==extreme)[-1]
        split='purged'
        if ev['t0']+pd.Timedelta(hours=84)<b.SPLIT:split='train' if ev['coin'] not in held else 'unseen_first_half'
        if ev['t0']>=b.SPLIT:split='test' if ev['coin'] not in held else 'unseen_test'
        for j in range(73):
            z=i+j
            if z>=len(d)-1:break
            row=d.iloc[z];ts=d.index[z]
            if not np.isfinite(row[field]):break
            if direction*(row[field]-extreme)>=0:extreme=float(row[field]);last=z
            if not row.features_ok or not np.isfinite(row.oi) or row.oi<=0:continue
            pause=z-last>=1;oi=row.oi3<=-.03
            hot=row.fmax24 if direction==1 else -row.fmin24
            cool=direction*row.fmean1
            tr=row.taker60 if direction==1 else 1/row.taker60
            taker=np.isfinite(tr) and tr<=1
            fund=hot>=.0003 and cool<=.0001
            br=float(btc.ret168.loc[ts]);rg='up' if br>=.03 else 'down' if br<=-.03 else 'sideways'
            rows.append(dict(event_id=ev['event_id'],coin=ev['coin'],event_t0=ev['t0'],decision_time=ts,
                i=z,delay_hours=j,direction=direction,q=-direction,extreme=extreme,atr=float(row.atr),
                full_window=ev['full_window'],split=split,event_regime=ev['regime'],decision_regime=rg,
                close=float(row.c),atr_pct=float(row.atr/row.c),ret24_abs=abs(float(row.ret24)),
                ret168_abs=abs(float(row.ret168)),btc_ret7=br,
                distance_to_extreme=float(direction*(extreme-row.c)/row.c),
                directional_ret1=float(direction*(row.c/d.c.iloc[z-1]-1)),
                price_pause_1h=int(pause),oi_drop_3pct_3h=int(oi),price_plus_oi=int(pause and oi),
                funding_cool=int(fund),taker_reversal_1h=int(taker),price_oi_taker=int(pause and oi and taker),
                week=int(ev['t0'].timestamp())//(7*86400)))
    panel=pd.DataFrame(rows);panel.index.name='landmark_id'
    return panel

def funding_cumulative(x, times_ns):
    idx=np.searchsorted(x['fund_ns'],times_ns,side='right')-1
    return np.where(idx>=0,x['fund_cum'][np.maximum(idx,0)],0)

def path_for(lm,x,res,horizon):
    if res not in x['arrays'] or horizon%res:return None
    a=x['arrays'][res];step=pd.Timedelta(minutes=res).value;t=pd.Timestamp(lm['decision_time']).value
    start=np.searchsorted(a['time_ns'],t+step);n=horizon//res
    if start+n>len(a['time_ns']):return None
    times=a['time_ns'][start:start+n]
    if not np.array_equal(times,t+step*np.arange(1,n+1)):return None
    q=int(lm['q']);entry=float(a['o'][start])*(1+q*b.SLIP)
    stop=max(lm['extreme']*1.002,entry+1.5*lm['atr']) if q<0 else min(lm['extreme']*.998,entry-1.5*lm['atr'])
    risk=abs(stop-entry)
    if not np.isfinite(risk) or risk<=0 or stop<=0:return None
    oh={k:a[k][start:start+n] for k in ['o','h','l','c']}
    if not all(np.isfinite(v).all() for v in oh.values()):return None
    fund=-q*(funding_cumulative(x,times-step)-funding_cumulative(x,np.array([t]))[0])
    favorable=oh['h'] if q>0 else oh['l'];adverse=oh['l'] if q>0 else oh['h']
    favorable_fill=favorable*(1-q*b.SLIP)
    net_best=q*(favorable_fill-entry)-b.RATE*(entry+favorable_fill)+fund
    gross_adverse=-q*(adverse-entry)
    continuation=bool((oh['h'].max()>lm['extreme']) if q<0 else (oh['l'].min()<lm['extreme']))
    mfe=float(net_best.max());mae=float(max(0,gross_adverse.max()))
    return dict(entry=entry,stop=stop,risk=risk,q=q,fund=fund,ohlc=oh,
                mfe_net_R=mfe/risk,mfe_net_pct=mfe/entry,mfe_net_atr=mfe/lm['atr'],
                mae_R=mae/risk,mae_pct=mae/entry,mae_atr=mae/lm['atr'],further_extreme=int(continuation))

def exit_net(path,k,raw_price):
    q=path['q'];px=raw_price*(1-q*b.SLIP);entry=path['entry']
    profit=q*(px-entry)-b.RATE*(entry+px)+path['fund'][k]
    return float(profit/path['risk']),float(profit/entry),float(px)

def target_trade(path,mode,target,optimistic=False):
    q=path['q'];entry=path['entry'];goal=(target/10000*entry) if mode=='bps' else target*path['risk']
    a=path['ohlc'];stop=path['stop'];dual=False
    for k in range(len(a['o'])):
        target_fill=(goal-path['fund'][k]+(q+b.RATE)*entry)/(q-b.RATE)
        raw_target=target_fill/(1-q*b.SLIP)
        open_stop=a['o'][k]<=stop if q>0 else a['o'][k]>=stop
        open_target=a['o'][k]>=raw_target if q>0 else a['o'][k]<=raw_target
        hit_stop=a['l'][k]<=stop if q>0 else a['h'][k]>=stop
        hit_target=a['h'][k]>=raw_target if q>0 else a['l'][k]<=raw_target
        if open_stop:reason='stop';raw=float(a['o'][k])
        elif open_target:reason='target';raw=float(raw_target)
        elif hit_stop and hit_target:
            dual=True;reason='target' if optimistic else 'stop';raw=float(raw_target if optimistic else stop)
        elif hit_stop:reason='stop';raw=float(stop)
        elif hit_target:reason='target';raw=float(raw_target)
        elif k==len(a['o'])-1:reason='timeout';raw=float(a['c'][k])
        else:continue
        r,pct,px=exit_net(path,k,raw)
        return dict(r=r,pct=pct,hit=int(reason=='target'),positive=int(r>0),reason=reason,bar=k,exit_price=px,dual=int(dual))
    raise AssertionError('No exit')

def outcome_row(lm,x,res,horizon):
    path=path_for(lm,x,res,horizon)
    if path is None:return None
    record={k:path[k] for k in ['entry','stop','risk','mfe_net_R','mfe_net_pct','mfe_net_atr','mae_R','mae_pct','mae_atr','further_extreme']}
    for name,mode,value in TARGETS:
        pessimistic=target_trade(path,mode,value);optimistic=target_trade(path,mode,value,True)
        for prefix,result in [('lo',pessimistic),('hi',optimistic)]:
            for field in ['r','pct','hit','positive','reason','bar','dual']:record[f'{name}_{prefix}_{field}']=result[field]
    return record

def calc_outcomes(panel,data):
    outcomes=[]
    for ix,lm in panel.iterrows():
        for res,hs in [(60,C['hourly_horizons_minutes']),(5,C['five_minute_horizons_minutes']),(1,C['one_minute_horizons_minutes'])]:
            for horizon in hs:
                result=outcome_row(lm,data[lm.coin],res,horizon)
                if result is None:continue
                result.update(landmark_id=int(ix),resolution_minutes=res,horizon_minutes=horizon)
                outcomes.append(result)
        if ix%500==0:print('OUTCOMES',ix,'/',len(panel),flush=True)
    return pd.DataFrame(outcomes)

def stats_for(vals,weeks,net_return=False):
    vals=np.asarray(vals,float);s=b.cluster_stats(vals,np.asarray(weeks));note='cluster_t_approximation'
    # A capped-profit sample with no losses cannot estimate an unobserved loss tail.
    if net_return and len(vals) and ((vals>0).all() or (vals<0).all() or np.std(vals)<1e-12):
        note='one_sided_or_constant_sample_tail_unobserved'
        s.update(lo=np.nan,hi=np.nan,se=np.nan,p=np.nan)
    return dict(n=s['n'],mean=s['mean'],ci95_low=s['lo'],ci95_high=s['hi'],se=s['se'],week_clusters=s['clusters'],p_positive=s['p'],inference_note=note)

def mask_split(df,name):
    if name=='all':return df.full_window.astype(bool)
    return df.full_window.astype(bool)&(df.split==name)

def strategy_summary(panel,outcomes):
    training_first=[]
    for _,g in panel[panel.full_window&(panel.split=='train')].groupby('event_id'):
        z=g[g.price_plus_oi==1].sort_values('delay_hours').head(1)
        if len(z):training_first.append(int(z.delay_hours.iloc[0]))
    median_delay=int(np.rint(np.median(training_first))) if training_first else None
    (OUT/'frozen_delay_control.json').write_text(json.dumps(dict(training_events=len(training_first),
       rounded_median_hours=median_delay,chosen_without_outcome_values=True),indent=2))
    first=[]
    for eid,g in panel[panel.full_window].groupby('event_id',sort=False):
        for rule in C['rules']:
            if rule=='immediate':z=g[g.delay_hours==0]
            elif rule=='wait_trainmedian_priceOI':z=g[g.delay_hours==median_delay] if median_delay is not None else g.iloc[:0]
            elif rule.startswith('wait_'):z=g[g.delay_hours==int(rule.split('_')[1][:-1])]
            else:z=g[g[rule]==1].sort_values('delay_hours').head(1)
            if z.empty:continue
            ix=z.index[0];first.append(dict(landmark_id=ix,rule=rule,event_id=eid))
    first=pd.DataFrame(first)
    selected=first.merge(outcomes,on='landmark_id').merge(panel.reset_index(),on=['landmark_id','event_id'])
    first[['rule','event_id','landmark_id']].to_csv(OUT/'first_signals.csv',index=False)
    records=[];paired=[];trade_records=[]
    for (rule,res,horizon),g in selected.groupby(['rule','resolution_minutes','horizon_minutes']):
        for split in ['all','train','test','unseen_test']:
            z=g[mask_split(g,split)]
            for name,mode,value in TARGETS:
                for bound in ['lo','hi']:
                    s=stats_for(z[f'{name}_{bound}_r'],z.week,net_return=True)
                    n=s['n'];ph=float(z[f'{name}_{bound}_hit'].mean());zz=1.959963984540054
                    if n:
                        center=(ph+zz*zz/(2*n))/(1+zz*zz/n)
                        half=zz*math.sqrt(ph*(1-ph)/n+zz*zz/(4*n*n))/(1+zz*zz/n)
                        wilson_lo=max(0,center-half);wilson_hi=min(1,center+half)
                    else:wilson_lo=wilson_hi=np.nan
                    records.append(dict(rule=rule,resolution_minutes=res,horizon_minutes=horizon,split=split,target=name,
                       bound=bound,n=s['n'],events=s['n'],expectancy_R=s['mean'],ci95_low_R=s['ci95_low'],ci95_high_R=s['ci95_high'],
                       target_rate=float(z[f'{name}_{bound}_hit'].mean()),profit_rate=float(z[f'{name}_{bound}_positive'].mean()),
                       target_wilson95_low_iid=wilson_lo,target_wilson95_high_iid=wilson_hi,
                       mean_net_bps=float(z[f'{name}_{bound}_pct'].mean()*10000),
                       mean_mae_pct=float(z.mae_pct.mean()),mean_mae_atr=float(z.mae_atr.mean()),
                       mean_initial_stop_pct=float((z.risk/z.entry).mean()),median_initial_stop_pct=float((z.risk/z.entry).median()),
                       mean_mfe_net_pct=float(z.mfe_net_pct.mean()),extreme_update_rate=float(z.further_extreme.mean()),
                       mean_delay_hours=float(z.delay_hours.mean()),ambiguous_rate=float(z[f'{name}_{bound}_dual'].mean()),
                       p_positive=s['p_positive'],p_bonf_strategy=min(1,s['p_positive']*(len(C['rules'])*len(TARGETS)*13)) if bound=='lo' and np.isfinite(s['p_positive']) else np.nan,
                       inference_note=s['inference_note'],
                       p_bonf_with_prior_search=min(1,s['p_positive']*(b.FAMILIES+len(C['rules'])*len(TARGETS)*13)) if bound=='lo' and np.isfinite(s['p_positive']) else np.nan,
                       week_clusters=s['week_clusters']))
        for name,_,_ in TARGETS:
            for bound in ['lo','hi']:
                cols=['event_id','coin','decision_time','split','week','rule','resolution_minutes','horizon_minutes','delay_hours']
                z=g[cols].copy();z['target']=name;z['bound']=bound
                for key in ['r','pct','hit','positive','reason','bar','dual']:z[key]=g[f'{name}_{bound}_{key}'].to_numpy()
                trade_records.append(z)
    # Paired same-event comparisons: do not treat absent signals as successful trades.
    for (res,horizon),allg in selected.groupby(['resolution_minutes','horizon_minutes']):
        control=allg[allg.rule=='immediate']
        for rule in C['rules'][1:]:
            case=allg[allg.rule==rule]
            m=case.merge(control,on='event_id',suffixes=('_case','_base'))
            for split in ['all','train','test','unseen_test']:
                z=m if split=='all' else m[m.split_case==split]
                for metric in ['mae_pct','mae_atr','further_extreme','mfe_net_pct','net10bp_lo_hit','net10bp_lo_r','R05_lo_hit']:
                    vals=z[metric+'_case']-z[metric+'_base'];s=stats_for(vals,z.week_case)
                    paired.append(dict(rule=rule,resolution_minutes=res,horizon_minutes=horizon,split=split,metric=metric,
                        n=s['n'],difference=s['mean'],ci95_low=s['ci95_low'],ci95_high=s['ci95_high'],
                        case_mean=float(z[metric+'_case'].mean()),base_mean=float(z[metric+'_base'].mean()),week_clusters=s['week_clusters']))
    pd.DataFrame(records).to_csv(OUT/'first_entry_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_event_comparisons.csv',index=False)
    clock=[]
    for (res,horizon),allg in selected.groupby(['resolution_minutes','horizon_minutes']):
        control=allg[allg.rule=='wait_trainmedian_priceOI']
        for rule in ['price_plus_oi','oi_drop_3pct_3h']:
            m=allg[allg.rule==rule].merge(control,on='event_id',suffixes=('_case','_base'))
            for split in ['all','train','test','unseen_test']:
                z=m if split=='all' else m[m.split_case==split]
                for metric in ['mae_pct','mae_atr','further_extreme','mfe_net_pct','net10bp_lo_hit','net10bp_lo_r','R05_lo_hit']:
                    s=stats_for(z[metric+'_case']-z[metric+'_base'],z.week_case)
                    clock.append(dict(rule=rule,control='wait_trainmedian_priceOI',control_delay_h=median_delay,
                        resolution_minutes=res,horizon_minutes=horizon,split=split,metric=metric,n=s['n'],
                        difference=s['mean'],ci95_low=s['ci95_low'],ci95_high=s['ci95_high'],
                        case_mean=float(z[metric+'_case'].mean()),base_mean=float(z[metric+'_base'].mean())))
    pd.DataFrame(clock).to_csv(OUT/'paired_frozen_clock_comparisons.csv',index=False)
    trades=pd.concat(trade_records,ignore_index=True)
    trades.decision_time=trades.decision_time.map(b.dubai);trades.to_csv(OUT/'fixed_rule_trades.csv',index=False)
    return selected

def regression(panel,outcomes,nonlinear=False):
    # Within-event FE + elapsed-hour FE distinguish indicator state from simply waiting.
    results=[]
    o=outcomes[outcomes.resolution_minutes==60].copy()
    merged=o.merge(panel.reset_index(),on='landmark_id');merged=merged[merged.full_window]
    covars=['atr_pct','ret24_abs','ret168_abs','btc_ret7','distance_to_extreme','directional_ret1']
    for split in ['all','train','test','unseen_test']:
        for horizon in C['hourly_horizons_minutes']:
            d=merged[(merged.horizon_minutes==horizon)&mask_split(merged,split)].copy()
            if d.empty:continue
            endpoints=pd.DataFrame(dict(further_extreme=d.further_extreme,mae_atr=d.mae_atr,
                mfe_net_atr=d.mfe_net_atr,hit_net10bp=d.net10bp_lo_hit,
                net10bp_R=d.net10bp_lo_r,hit_R05=d.R05_lo_hit),index=d.index)
            arr=d[covars].to_numpy(float);std=arr.std(axis=0);std[std<1e-12]=1
            arr=(arr-arr.mean(axis=0))/std
            controls=np.column_stack([np.ones(len(d)),arr,
                pd.get_dummies(d.event_id,drop_first=True,dtype=float).to_numpy(),
                pd.get_dummies(d.delay_hours,drop_first=True,dtype=float).to_numpy()])
            if nonlinear:
                # Check whether pause merely identifies prices already far from the peak.
                distance=d.distance_to_extreme.to_numpy()/d.atr_pct.to_numpy()
                bins=pd.cut(distance,[-np.inf,0,.25,.5,1,2,4,8,np.inf],include_lowest=True)
                velocity=d.directional_ret1.to_numpy()/d.atr_pct.to_numpy()
                extra=np.column_stack([pd.get_dummies(bins,drop_first=True,dtype=float).to_numpy(),velocity,velocity**2])
                scale=extra.std(axis=0);scale[scale<1e-12]=1
                extra=(extra-extra.mean(axis=0))/scale
                controls=np.column_stack([controls,extra])
            # Each event has equal total weight; repeated landmarks are not independent trades.
            weight=1/d.groupby('event_id').event_id.transform('count').to_numpy()
            sw=np.sqrt(weight);x=controls*sw[:,None]
            s=d[SIGNALS].to_numpy(float)*sw[:,None];y=endpoints.to_numpy(float)*sw[:,None]
            fit,_,rank,_=np.linalg.lstsq(x,np.column_stack([s,y]),rcond=1e-10)
            resid=np.column_stack([s,y])-x@fit;sr=resid[:,:len(SIGNALS)];yr=resid[:,len(SIGNALS):]
            _,groups=np.unique(d.week,return_inverse=True);ng=groups.max()+1
            for k,signal in enumerate(SIGNALS):
                xx=sr[:,k];den=float(xx@xx)
                support_events=int(d[d[signal]==1].event_id.nunique())
                if den<1e-10:continue
                for m,metric in enumerate(METRICS):
                    beta=float(xx@yr[:,m]/den);err=yr[:,m]-beta*xx
                    scores=np.bincount(groups,weights=xx*err)
                    if ng<2 or len(d)<=rank+1:se=np.nan
                    else:se=math.sqrt(ng/(ng-1)*(len(d)-1)/(len(d)-rank-1)*float(scores@scores))/den
                    critical=stats.t.ppf(.975,ng-1) if ng>=2 else np.nan
                    p=float(2*stats.t.sf(abs(beta/se),ng-1)) if se>0 else np.nan
                    primary=signal==C['primary_rule'] and horizon==C['primary_horizon_minutes'] and metric in C['primary_endpoints']
                    results.append(dict(split=split,horizon_minutes=horizon,signal=signal,metric=metric,
                        beta=beta,ci95_low=beta-critical*se,ci95_high=beta+critical*se,p_two_sided=p,
                        p_bonf_144=min(1,p*144) if np.isfinite(p) else np.nan,
                        primary_endpoint=primary,p_bonf_primary4=min(1,p*4) if primary and np.isfinite(p) else np.nan,
                        landmarks=len(d),events=int(d.event_id.nunique()),signal_events=support_events,week_clusters=ng,
                        estimable_within_event_variance=den))
            print('REGRESSION',split,horizon,'events',d.event_id.nunique(),flush=True)
    pd.DataFrame(results).to_csv(OUT/('adjusted_associations_nonlinear.csv' if nonlinear else 'adjusted_associations.csv'),index=False)

def panel_descriptive(panel,outcomes):
    d=outcomes.merge(panel.reset_index(),on='landmark_id');d=d[d.full_window]
    records=[]
    for (res,horizon),g in d.groupby(['resolution_minutes','horizon_minutes']):
        for signal in SIGNALS:
            for state in [0,1]:
                for split in ['all','test','unseen_test']:
                    z=g[(g[signal]==state)&mask_split(g,split)]
                    # Average within each episode first; every episode has equal weight.
                    for metric in ['mae_pct','mae_atr','mfe_net_pct','further_extreme','net10bp_lo_hit','net10bp_lo_r','R05_lo_hit']:
                        event=z.groupby('event_id').agg(value=(metric,'mean'),week=('week','first'))
                        s=stats_for(event.value,event.week)
                        records.append(dict(resolution_minutes=res,horizon_minutes=horizon,signal=signal,state=state,
                           split=split,metric=metric,landmarks=len(z),events=len(event),mean=s['mean'],
                           ci95_low=s['ci95_low'],ci95_high=s['ci95_high'],week_clusters=s['week_clusters']))
    pd.DataFrame(records).to_csv(OUT/'landmark_episode_weighted.csv',index=False)

def main():
    data=load_all();btc=data['BTC']['hour'].ret168
    events=sum([scan_events(c,x,btc) for c,x in data.items()],[]);events=sorted(events,key=lambda e:(e['t0'],e['coin']))
    original=set(pd.read_csv(ROOT/'results/events.csv').event_id)
    assert original.issubset({e['event_id'] for e in events}), 'Original episodes missing in causal scan'
    evdf=pd.DataFrame(events);evdf.t0=evdf.t0.map(b.dubai);evdf.to_csv(OUT/'events.csv',index=False)
    print('EVENTS',len(events),'complete 84h',sum(e['full_window'] for e in events),flush=True)
    panel=make_landmarks(events,data);outcomes=calc_outcomes(panel,data)
    export=panel.copy()
    for col in ['event_t0','decision_time']:export[col]=export[col].map(b.dubai)
    export.to_csv(OUT/'landmarks.csv')
    outcomes.to_csv(OUT/'outcomes.csv',index=False)
    strategy_summary(panel,outcomes);panel_descriptive(panel,outcomes);regression(panel,outcomes);regression(panel,outcomes,True)
    coverage=outcomes.merge(panel[['event_id','full_window','split']],left_on='landmark_id',right_index=True)
    cv=[]
    for (res,horizon),g in coverage.groupby(['resolution_minutes','horizon_minutes']):
        complete=g[g.full_window]
        cv.append(dict(resolution_minutes=res,horizon_minutes=horizon,landmarks=len(g),
            independent_events=g.event_id.nunique(),primary_landmarks=len(complete),primary_independent_events=complete.event_id.nunique()))
    pd.DataFrame(cv).to_csv(OUT/'outcome_coverage.csv',index=False)
    audit=dict(original_events=len(original),causal_events=len(events),full_window_events=sum(e['full_window'] for e in events),
       landmarks=len(panel),outcome_rows=len(outcomes),same_data_followup=True,
       one_minute_has_no_six_month_history=True,per_rule_event_entries_max=1,
       target_definitions='net of fixed fees/slippage and oracle-proxy funding',
       ambiguity_bounds='bar-open gaps observed first; SL/TP both inside bar provide lower/upper bounds')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print('DONE',json.dumps(audit),flush=True)

if __name__=='__main__':main()
