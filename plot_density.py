from plotnine import *
import pandas as pd

# Assuming df_plot is already created as before
summary_df = (df_plot.groupby('groups')
             .agg({
                 'y_pred': 'first',
                 'y_lower': 'first',
                 'y_upper': 'first'
             })
             .reset_index())

(ggplot(df_plot, aes(x='y', fill='groups'))
 + geom_density(alpha=0.7)
 + geom_vline(aes(xintercept='y_pred'),
              data=summary_df,
              color='darkred',
              size=1)
 + geom_vline(aes(xintercept='y_lower'),
              data=summary_df,
              linetype='dashed',
              alpha=0.5)
 + geom_vline(aes(xintercept='y_upper'),
              data=summary_df,
              linetype='dashed',
              alpha=0.5)
 + facet_grid('groups ~ .')
 + labs(x='Log House Price ($)', y='')
 + theme_minimal()
 + theme(
     strip_text_y=element_blank(),
     axis_text_y=element_blank(),
     axis_ticks_y=element_blank(),
     panel_spacing=-0.25,
     legend_position='none'
 ))