"""Build a readable Japanese report and a compact CSV from audited outputs."""
import json
import pandas as pd
import reference_study as r

def main():
    s=pd.read_csv(r.OUT/'first_entry_summary.csv')
    primary=(s.resolution_minutes==60)&(s.horizon_minutes==60)&(s.split=='test')&(s.bound=='lo')&s.target.isin(['net1bp','net10bp'])
    fine=(s.resolution_minutes==5)&(s.horizon_minutes==30)&(s.split=='all')&(s.target=='net10bp')&(s.bound=='lo')
    late=(s.rule.isin(['price_plus_oi','oi_drop_3pct_3h','wait_trainmedian_priceOI']))&(s.resolution_minutes==60)&(s.horizon_minutes==720)&(s.split=='test')&(s.target=='net10bp')&(s.bound=='lo')
    cols=['rule','resolution_minutes','horizon_minutes','target','split','n','target_rate',
          'target_wilson95_low_iid','target_wilson95_high_iid','expectancy_R','ci95_low_R','ci95_high_R',
          'mean_net_bps','mean_initial_stop_pct','mean_delay_hours','ambiguous_rate','inference_note']
    s[primary|fine|late][cols].to_csv(r.ROOT/'HL_reference_indicator_summary_2026-10-12.csv',index=False)
    a=pd.read_csv(r.OUT/'adjusted_associations_nonlinear.csv')
    assoc=a[(a.split=='test')&(a.horizon_minutes==60)&a.signal.isin(['price_pause_1h','oi_drop_3pct_3h','price_plus_oi'])]
    pair=pd.read_csv(r.OUT/'paired_event_comparisons.csv')
    pm=pair[(pair.rule=='price_plus_oi')&(pair.split=='test')&(pair.resolution_minutes==60)&(pair.horizon_minutes==60)]
    clocks=pd.read_csv(r.OUT/'paired_frozen_clock_comparisons.csv')
    cm=clocks[(clocks.rule=='price_plus_oi')&(clocks.split=='test')&(clocks.resolution_minutes==60)&(clocks.horizon_minutes==60)]
    focus=['immediate','price_pause_1h','oi_drop_3pct_3h','price_plus_oi','wait_trainmedian_priceOI']
    table=s[primary&s.rule.isin(focus)][['rule','target','n','target_rate','expectancy_R','mean_net_bps','mean_initial_stop_pct']]
    lines=['Hyperliquid 参考指標と小幅利確の検証',
       '期間：2026-03-25 04:00〜2026-10-12 04:00、ドバイ時間。',
       '主対象76イベント。前回データの追検証であり、新しい独立期間外検証ではない。',
       '指標判定は毎時。1時間足の主解析、5分足・1分足の短期補足を分ける。',
       '以下の時間は急変検知からではなく、条件を満たしてエントリーした後の時間。',
       '', '結果の要点：',
       '・小幅利確の機会は多かった。到達率だけで、損失込みの平均利益は決まらない。',
       '・後半の即時逆張りは、1hでネット0.01%目標に28/30件到達。しかし未到達2件の平均損失は約1.62%、全体平均は約-0.098%。',
       '・OI3h3%減少後は、1hでネット0.1%へ15/16件、12hで16/16件到達。',
       '・価格停滞＋OIでは、同目標に1hで13/16件、12hで16/16件到達。',
       '・同じ16イベントの無条件対照や、学習側だけで固定した8h待機対照に対する改善は、信頼区間がゼロをまたぐ。',
       '・価格停滞は、その後1hの極値更新が少ない方向に関連。価格位置の非線形調整後、後半推定-11.5ポイント、95%区間-20.0〜-2.9ポイント。ただし144比較補正後は有意ではなく、モデル依存。',
       '・価格停滞とOIの組合せが、逆行幅や短期利確率を確実に改善する証拠は残っていない。',
       '・funding冷却と成行比率の追加情報は今回確立していない。',
       '', '後半・1h以内の固定利確（原則同時SL/TPはSL先の下限）：',table.to_string(index=False),
       '', '12h以内の小幅利確：',s[late][['rule','n','target_rate','target_wilson95_low_iid','target_wilson95_high_iid','expectancy_R','mean_net_bps','mean_initial_stop_pct']].to_string(index=False),
       '', '価格停滞＋OIと即時入口：同イベント比較',pm[['metric','n','case_mean','base_mean','difference','ci95_low','ci95_high']].to_string(index=False),
       '', '価格停滞＋OIと8h単純待機：同イベント比較',cm[['metric','n','case_mean','base_mean','difference','ci95_low','ci95_high']].to_string(index=False),
       '', '他の情報を揃えた1h結果：',assoc[['signal','metric','beta','ci95_low','ci95_high','p_bonf_144','events']].to_string(index=False),
       '', '30分・5分足の短期補足：',s[fine&s.rule.isin(focus)][['rule','n','target_rate','expectancy_R','mean_net_bps']].to_string(index=False),
       '', '1分足：主対象は独立3イベントのみ。ルール最初の入口で評価可能なのは1〜2件。結論を出せない。',
       '', '解釈と制約：',
       '参考候補として残すなら、価格停滞を状態把握に、OI減少を小反発の補助観察に用いる。売買の自動許可閾値としては確定していない。',
       '損切りは前回の既知極値とATRの遠い方を維持。価格＋OIの後半初期ストップは平均約7.7%、中央値約4.8%。ネット0.1%の利確に対して大きい。',
       '全16件で利確しても、損失の尾部を観測していないため、平均Rのt信頼区間やp値は表示しない。独立仮定のWilson勝率区間は約81〜100%だが、銘柄間の連動を反映しない参考値。',
       '最大逆行・最大有利変動は固定時間の価格経路の診断値で、利確後の価格も含む。実際に取れた利益として扱わない。',
       '週クラスター区間、固定費用・スリッページ、履歴oracle代理、1h内の約定順序、現在の候補銘柄選択に制約が残る。',
       '詳細定義、全結果、元データ、コードはREADMEとreference_resultsを参照。']
    (r.ROOT/'report_ja.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('Report and compact CSV written.')

if __name__=='__main__':main()
