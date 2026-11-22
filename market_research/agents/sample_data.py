"""Clearly labelled, fixed, illustrative data. Used only when "sample data" mode is on."""
from ..models import Finding, Research

SAMPLE_TITLE = "SAMPLE DATA (illustrative, not a real source)"

SAMPLE_RESEARCH = Research(
    mode="sample",
    provider="sample",
    findings=[
        Finding(id="F1", claim="SAMPLE: Foodmandu is one of the earliest and best-known food delivery platforms in Nepal, operating mainly in Kathmandu Valley.", source_title=SAMPLE_TITLE),
        Finding(id="F2", claim="SAMPLE: Pathao, a ride-hailing and logistics app, also offers food delivery in Nepal.", source_title=SAMPLE_TITLE),
        Finding(id="F3", claim="SAMPLE: Delivery demand is concentrated in Kathmandu Valley and other large urban areas.", source_title=SAMPLE_TITLE),
        Finding(id="F4", claim="SAMPLE: Platforms typically earn through restaurant commissions and delivery fees.", source_title=SAMPLE_TITLE),
        Finding(id="F5", claim="SAMPLE: Common challenges include traffic congestion, delivery-partner costs and thin margins.", source_title=SAMPLE_TITLE),
    ],
    data_gaps=[
        "Sample data is illustrative and unverified; no market size or growth figures are included.",
        "No market share data for any company.",
    ],
)
