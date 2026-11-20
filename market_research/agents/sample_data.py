"""Clearly labelled, fixed, illustrative data. Used only when "sample data" mode is on."""
from ..models import Finding, Research

SAMPLE_RESEARCH = Research(
    mode="sample",
    findings=[
        Finding(claim="SAMPLE: Foodmandu is one of the earliest and best-known food delivery platforms in Nepal, operating mainly in Kathmandu Valley."),
        Finding(claim="SAMPLE: Pathao, a ride-hailing and logistics app, also offers food delivery in Nepal."),
        Finding(claim="SAMPLE: Delivery demand is concentrated in Kathmandu Valley and other large urban areas."),
        Finding(claim="SAMPLE: Platforms typically earn through restaurant commissions and delivery fees."),
        Finding(claim="SAMPLE: Common challenges include traffic congestion, delivery-partner costs and thin margins."),
    ],
    data_gaps=[
        "Sample data is illustrative and unverified; no market size or growth figures are included.",
        "No market share data for any company.",
    ],
)
