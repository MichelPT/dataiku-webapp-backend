# In your services/dns_dnsv_plot.py or equivalent file

from plotly.subplots import make_subplots
# Ensure all your helper functions are imported from your plotting service
from services.plotting_service import (
    main_plot
)


def plot_dns_dnsv(df, title='DNS + DNSV Analysis'):
    """
    Creates a comprehensive DNS-DNSV plot based on the working Colab logic.
    """
    sequence = ['MARKER', 'GR', 'RT', 'NPHI_RHOB', 'DNS', 'DNSV']
    fig = main_plot(df, sequence, title=title)

    return fig
