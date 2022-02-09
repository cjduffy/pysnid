import numpy as np

def get_ciel(value, n=5):
    """ """
    return (1+value//n)*n


def get_polartwin(ax):
    """ """
    
    ax2 = ax.figure.add_axes(ax.get_position(), projection='polar', 
                             label='twin', frameon=False,
                             theta_direction=ax.get_theta_direction(),
                             theta_offset=ax.get_theta_offset())
    ax2.xaxis.set_visible(False)

    # There should be a method for this, but there isn't... Pull request?
    ax2._r_label_position._t = (22.5 + 180, 0.0)
    ax2._r_label_position.invalidate()
    return ax2
    
def make_spiderplot( values, 
                      ax=None, rtwin_from=None, labels=None, title=None, gcolor=None, 
                      facecolor="None", edgecolor="C0", 
                      rlabel_angle=None, lw=2, alpha=None, rlabel=None, 
                      highlight=None, highlight_color="k", highlight_lw=1,
                      rlabel_va="center", rlabel_ha="left", rlabel_rotation=None,
                      gridn=4, nticks=None,
                     **kwargs):
    """ """
    import matplotlib.pyplot as mpl
    from matplotlib.colors import to_rgba
        
    nparams = len(values)
    if labels is None:
        labels = np.arange(nparams)

    # - values
    values = np.concatenate((values,[values[0]]))
    # - angles
    angles = np.linspace(0, 2*np.pi, nparams, endpoint=False)
    angles = np.concatenate((angles,[angles[0]]))

    # --> Figures
    if rtwin_from is not None:
        ax = get_polartwin(rtwin_from)
        fig = ax.figure
        nticks = len(rtwin_from.get_yticks())+2
    elif ax is None:
        fig = mpl.figure(figsize=[4,4])
        ax = fig.add_axes([0.1,0.1,0.8,0.8], polar=True)
    else:
        fig = ax.figure
        
    ax.fill(angles, values, facecolor=facecolor, edgecolor=edgecolor, 
            lw=lw, alpha=alpha, **kwargs)
    
    if highlight_color == "None":
        highlight = None
    if highlight is not None:
        angles_ = np.linspace(0,360, 100)*np.pi/180
        line_ = 0*angles_ + highlight
        ax.plot(angles_, line_, 
                color = highlight_color, lw=highlight_lw)
        
    #
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    #
#    ax.set_yticks(markers)
    if title is not None:
        ax.set_title(name)
        
    ceil_top = get_ciel(np.max(values), n=gridn)
    
    if nticks is None:
        nticks = np.max([4,int(ceil_top//gridn+1)])
        if nticks>7:
            gridn *= 2
            ceil_top = get_ciel(np.max(values), n=gridn)
            nticks = np.max([4,int(ceil_top//(gridn)+1)])
        
    ax.set_rlim(0, ceil_top)
    ax.set_rticks(np.linspace(0, ceil_top, nticks)[1:-1])

    if gcolor == "None":
        ax.grid(False)
        gcolor = "r" # just to make sure we don't see it.
    else:
        ax.grid(True)
        
    if gcolor is None:
        gcolor = to_rgba("0.8",0.1)

    ax.tick_params(axis="y", labelsize="small", grid_color=gcolor, 
                   labelcolor=to_rgba(edgecolor, alpha=alpha), zorder=1)
    ax.tick_params(axis="x", labelsize="small", grid_color="k",
                  labelcolor="0.5", zorder=2)
    if rlabel_angle is not None:
        ax.set_rlabel_position(rlabel_angle)  # Move radial labels away from plotted line
    
    if rlabel is not None:
        label_pos = ax.get_rlabel_position()
        
        ax.text(label_pos*np.pi/180, ceil_top*1.1, rlabel, 
                va=rlabel_va, 
                ha=rlabel_ha,
                rotation=90-label_pos if rlabel_rotation else rlabel_rotation,
                color=edgecolor, fontsize="small")
        
    return fig
