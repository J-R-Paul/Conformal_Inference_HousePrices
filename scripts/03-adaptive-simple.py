# %%
import os
import logging
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import train_test_split

from mapie.regression import MapieRegressor, MapieQuantileRegressor

from utils import load_prep_data, train_cal_test_split, prep_conditional_data, check_coverage, plot_conditional_intervals



# Only set up file logging if not running in interactive/notebook environment
if not hasattr(sys, 'ps1') and 'ipykernel' not in sys.modules:
    # Ensure the logs directory exists
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(filename=os.path.join(log_dir, '03-Adaptive-simple.log'), level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
else:
    # Set up basic console logging for interactive sessions
    logging.basicConfig(level=logging.CRITICAL, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
# %%
df = load_prep_data(polynomial=False)
X, y, groups  = prep_conditional_data(df)
X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(X, y, groups, test_size=0.3, random_state=123)


# %%
# Quantile Regression
q_estimator = QuantileRegressor(quantile=0.5, alpha=0)
mapie = MapieQuantileRegressor(q_estimator, method="quantile", cv='split', alpha=0.05)
mapie.fit(X_train, y_train, calib_size=0.3, random_state=26)
# %%
# Predictions
y_pred, y_pis = mapie.predict(X_test)

plot_conditional_intervals(y_test, y_pred, y_pis, group_test, title="Quantile Conformal Prediction Intervals")

# %%
# Adaptive
# ------------------------------------------------------------------------------
# Adaptive Normalised Score Functions 
# ------------------------------------------------------------------------------
from mapie.conformity_scores import ResidualNormalisedScore
base = LinearRegression(fit_intercept=False)
X_train_res, X_res_res, y_train_res, y_res_res = train_test_split(X_train, y_train, test_size=0.3, random_state=26)
base.fit(X_train_res, y_train_res)
residual_estimator = LinearRegression(fit_intercept=False)
residual_estimator.fit(X_res_res, np.abs(np.subtract(y_res_res, base.predict(X_res_res))))

mapie = MapieRegressor(base, cv='prefit',  
                       conformity_score=ResidualNormalisedScore(
                                        residual_estimator=residual_estimator, 
                                        prefit=True
                                        ))

mapie.fit(X_train, y_train, random_state=26)
y_pred, y_pis = mapie.predict(X_test, alpha=0.05)
# %%
plot_conditional_intervals(y_test, y_pred, y_pis, group_test, title="Adaptive Conformal Prediction Intervals")

# %%
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Sea born plots
df_plot = pd.DataFrame({"y": y_test,
                         "y_pred": y_pred,
                         "y_lower": y_pis[:, 0].ravel(),
                          "y_upper": y_pis[:, 1].ravel(),
                          "groups" : group_test
                          })

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(len(df_plot['groups'].unique()), rot=-.25, light=.7)
g = sns.FacetGrid(df_plot, row="groups", hue="groups", aspect=15, height=.9, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "y",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "y", clip_on=False, color="w", lw=2, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "y")

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.05)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

plt.show()
# %%

# %%
df_plot = pd.DataFrame({"y": y_test,
                         "y_pred": y_pred,
                         "y_lower": y_pis[:, 0].ravel(),
                          "y_upper": y_pis[:, 1].ravel(),
                          "groups" : group_test
                          })
# First, let's create a summary dataframe with mean values per group
group_summary = df_plot.groupby('groups').agg({
    'y_pred': 'first',  # since it's the same for each group
    'y_lower': 'first',
    'y_upper': 'first'
}).reset_index()

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(len(df_plot['groups'].unique()), rot=-.25, light=.7)
g = sns.FacetGrid(df_plot, row="groups", hue="groups", aspect=15, height=.5, palette=pal)

# Draw the densities
g.map(sns.kdeplot, "y",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "y", clip_on=False, color="w", lw=2, bw_adjust=.5)

# Add mean and CI lines
for ax, (name, group_data) in zip(g.axes.flat, group_summary.iterrows()):
    color = pal[name] if isinstance(name, int) else pal[list(df_plot['groups'].unique()).index(group_data['groups'])]
    
    # Add mean line
    ax.axvline(x=group_data['y_pred'], color="darkred", linestyle='-', linewidth=2)
    
    # Add CI lines
    ax.axvline(x=group_data['y_lower'], color=color, linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=group_data['y_upper'], color=color, linestyle='--', linewidth=1, alpha=0.5)
    
    # Optional: add shading between CI lines
    ax.axvspan(group_data['y_lower'], group_data['y_upper'], 
               color=color, alpha=0.1)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "y")

g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(xlabel="Log House Price ($)")
g.set(yticks=[], ylabel="")
g.despine(  bottom=True, left=True)
# plt.tight_layout()
plt.show()

# %%
import plotnine as pn
import pandas as pd

# Assuming df_plot is already created as before
summary_df = (df_plot.groupby('groups')
             .agg({
                 'y_pred': 'first',
                 'y_lower': 'first',
                 'y_upper': 'first'
             })
             .reset_index())

group_labels = {
    '4': '1-4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9-10'
}

(pn.ggplot(df_plot, pn.aes(x='y', fill='groups'))
 + pn.geom_density(alpha=0.7)
 + pn.geom_vline(pn.aes(xintercept='y_pred'),
              data=summary_df,
              color='darkred',
              size=1)
 + pn.geom_vline(pn.aes(xintercept='y_lower'),
              data=summary_df,
              linetype='dashed',
              alpha=0.5)
 + pn.geom_vline(pn.aes(xintercept='y_upper'),
              data=summary_df,
              linetype='dashed',
              alpha=0.5)
 + pn.facet_grid('groups ~ .', labeller=pn.as_labeller(group_labels))
 + pn.labs(x='Log House Price ($)', y='')
 + pn.theme_minimal()
 + pn.theme(
    axis_line=pn.element_blank(),
    panel_grid_major=pn.element_blank(),
    panel_grid_minor=pn.element_blank(),
    panel_border=pn.element_blank(),
    axis_text_y=pn.element_blank(),
    strip_text_y=pn.element_text(size=12),  # adjust size as needed
    strip_background=pn.element_blank(),  # removes the grey background,
    panel_spacing=-0.0,
    legend_position='none',
    strip_text=pn.element_text(angle=0)
 ))

# %%

# First, create separate rectangle data for the CI shading
rect_data = pd.DataFrame({
    'groups': summary_df['groups'],
    'y_lower': summary_df['y_lower'],
    'y_upper': summary_df['y_upper']
})


(pn.ggplot(df_plot, pn.aes(x='y', fill='groups'))
 # Add the CI shading using the rect_data
 + pn.geom_rect(data=rect_data,
                mapping=pn.aes(xmin='y_lower',
                              xmax='y_upper',
                              ymin=-float('inf'),
                              ymax=float('inf'),
                              fill='groups'),
                alpha=0.1,
                inherit_aes=False)  # Important: don't inherit aesthetics
 + pn.geom_density(alpha=0.7)
 + pn.geom_vline(pn.aes(xintercept='y_pred'),
              data=summary_df,
              color='darkred',
              size=1)
 + pn.geom_vline(pn.aes(xintercept='y_lower'),
              data=summary_df,
              linetype='dashed',
              alpha=0.5)
 + pn.geom_vline(pn.aes(xintercept='y_upper'),
              data=summary_df,
              linetype='dashed',
              alpha=0.5)
 + pn.facet_grid('groups ~ .', labeller=pn.as_labeller(group_labels))
 + pn.labs(x='Log House Price ($)', y='')
 + pn.theme_minimal()
 + pn.theme(
    axis_line=pn.element_blank(),
    panel_grid_major=pn.element_blank(),
    panel_grid_minor=pn.element_blank(),
    panel_border=pn.element_blank(),
    axis_text_y=pn.element_blank(),
    strip_text_y=pn.element_text(size=12),
    strip_background=pn.element_blank(),
    panel_spacing=-0.0,
    legend_position='none',
    strip_text=pn.element_text(angle=0)
 ))
# %%
