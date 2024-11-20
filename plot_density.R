
library(ggplot2)
library(dplyr)

# Assuming you have your data in a data frame with the same structure:
# df_plot = data.frame(
#   y = y_test,
#   y_pred = y_pred,
#   y_lower = y_pis[,1],
#   y_upper = y_pis[,2],
#   groups = group_test
# )

# Create the plot
ggplot(df_plot, aes(x = y, fill = groups)) +
  # Add density plots
  geom_density(alpha = 0.7) +
  
  # Add mean and CI lines using geom_vline
  geom_vline(data = df_plot %>% 
               group_by(groups) %>% 
               summarise(y_pred = first(y_pred),
                        y_lower = first(y_lower),
                        y_upper = first(y_upper)),
             aes(xintercept = y_pred),
             color = "darkred",
             linewidth = 1) +
  
  # Add CI lines
  geom_vline(data = df_plot %>% 
               group_by(groups) %>% 
               summarise(y_pred = first(y_pred),
                        y_lower = first(y_lower),
                        y_upper = first(y_upper)),
             aes(xintercept = y_lower),
             linetype = "dashed",
             alpha = 0.5) +
  geom_vline(data = df_plot %>% 
               group_by(groups) %>% 
               summarise(y_pred = first(y_pred),
                        y_lower = first(y_lower),
                        y_upper = first(y_upper)),
             aes(xintercept = y_upper),
             linetype = "dashed",
             alpha = 0.5) +
  
  # Facet by groups
  facet_grid(groups ~ ., switch = "y") +
  
  # Customize theme and labels
  labs(x = "Log House Price ($)", y = "") +
  theme_minimal() +
  theme(
    strip.text.y = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.spacing = unit(-0.25, "lines"),
    legend.position = "none"
  )