import unittest
import numpy as np
import reference_study as r

def path(opens,highs,lows,closes,stop=99.,q=1):
    e=100*(1+q*r.b.SLIP)
    return dict(q=q,entry=e,stop=stop,risk=abs(e-stop),fund=np.zeros(len(opens)),
       ohlc={k:np.array(v,float) for k,v in zip(['o','h','l','c'],[opens,highs,lows,closes])})

class NetProfitTests(unittest.TestCase):
    def test_all_winners_does_not_produce_certain_positive_expectancy(self):
        z=r.stats_for(np.full(16,.25),np.arange(16)//2,net_return=True)
        self.assertEqual(z['mean'],.25)
        self.assertTrue(np.isnan(z['p_positive']))
        self.assertTrue(np.isnan(z['ci95_low']))
    def test_gross_move_smaller_than_cost_is_not_a_win(self):
        p=path([100],[100.10],[99.99],[100.05])
        z=r.target_trade(p,'bps',1)
        self.assertEqual(z['hit'],0)
        self.assertLess(z['pct'],0)
    def test_net_target_is_after_both_fees_and_both_slippages(self):
        p=path([100],[100.4],[99.9],[100.3])
        z=r.target_trade(p,'bps',10)
        self.assertEqual(z['hit'],1)
        self.assertAlmostEqual(z['pct'],.001,places=12)
    def test_short_net_target_is_symmetric(self):
        p=path([100],[100.1],[99.6],[99.7],stop=101,q=-1)
        z=r.target_trade(p,'bps',10)
        self.assertEqual(z['hit'],1)
        self.assertAlmostEqual(z['pct'],.001,places=12)
    def test_ambiguous_same_bar_produces_bounds(self):
        p=path([100],[100.4],[98.9],[99.5])
        lo=r.target_trade(p,'bps',10);hi=r.target_trade(p,'bps',10,True)
        self.assertEqual(lo['reason'],'stop');self.assertEqual(hi['reason'],'target')
        self.assertTrue(lo['dual']);self.assertLess(lo['r'],hi['r'])
    def test_observed_open_profit_precedes_later_bar_stop(self):
        p=path([100,100.5],[100.1,101],[99.9,98],[100.05,99])
        lo=r.target_trade(p,'bps',10)
        self.assertEqual(lo['reason'],'target');self.assertFalse(lo['dual'])
    def test_observed_open_stop_precedes_later_target(self):
        p=path([100,98.5],[100.1,101],[99.9,98],[100.05,100])
        hi=r.target_trade(p,'bps',10,True)
        self.assertEqual(hi['reason'],'stop');self.assertLess(hi['r'],-1)

if __name__=='__main__':unittest.main()
