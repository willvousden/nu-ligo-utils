import numpy as np

class ProposalTracking:
    def __init__(self, filename):
        with open(filename, 'r') as inp:
            raw_header = inp.readline().split()
            header = [p for p in raw_header]
            dt = []
            params=[]
            for p in header:
                if p == 'proposal':
                    ptype = '|S64'
                elif p == 'accepted' or p == 'cycle':
                    ptype = 'int'
                else:
                    ptype = 'float'
                    if p.find('_p') < 0:
                        params.append(p)
                dt.append((p, ptype))

            self._tracking_array = np.genfromtxt(inp, dtype=dt)
            self._proposals = np.unique(self._tracking_array['proposal'])

    def plot(self, params, confidence_levels=[0.90, 0.95, 0.99], downsample=100):
        from plotutils import plotutils as pu
        from matplotlib import pyplot as plt
        from matplotlib.widgets import CheckButtons

        from itertools import cycle
        lines = ['solid', 'dashed', 'dashdot', 'dotted']
        linecycler = cycle(lines)

        N = len(self._tracking_array)
        tracking_array = self._tracking_array

        param1 = tracking_array[params[0]]
        param2 = tracking_array[params[1]]
        pu.plot_greedy_kde_interval_2d(np.vstack([param1, param2]).T, [0.90,0.95,0.99])
        ax = plt.gca()

        arrows = {}
        for proposal in self._proposals:
            ls = next(linecycler)
            sel = tracking_array['proposal'] == proposal
            accepted = tracking_array['accepted'][sel]
            xs = param1[sel]
            ys = param2[sel]
            x_ps = tracking_array[params[0]+'_p'][sel]
            y_ps = tracking_array[params[1]+'_p'][sel]
            dxs = x_ps - xs
            dys = y_ps - ys

            arrows[proposal] = []
            for x, y, dx, dy, a in zip(xs,ys,dxs,dys,accepted):
                if dx != 0 or dy != 0:
                    c = 'green' if a else 'red'
                    arrow = ax.arrow(x, y, dx, dy, fc=c, ec=c, alpha=0.5, visible=False, linestyle=ls)
                    arrows[proposal].append(arrow)

        plt.subplots_adjust(left=0.5)
        rax = plt.axes([0.05, 0.4, 0.4, 0.35])
        check = CheckButtons(rax, self._proposals, [False for i in self._proposals])

        def func(proposal):
            N = len(arrows[proposal])
            step = int(np.floor(N/float(downsample)))
            step = step if step is not 0 else 1
            for l in arrows[proposal][::step]:
                l.set_visible(not l.get_visible())
            plt.draw()

        check.on_clicked(func)
        plt.show()

    def proposals(self):
        print self._proposals
