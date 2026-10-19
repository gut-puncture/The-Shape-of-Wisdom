# Design Notes

The old paper compared a transient early event to persistent late events. That asymmetry made the lag vulnerable to a metric-definition critique.

The corrected operationalization is fully persistence-based:
- `d2`: first layer after which the eventual winner never leaves the top-2
- `d1`: first layer after which it never leaves top-1
- `d25`, `d50`, `d75`: first layer after which the winner gap never drops below 25%, 50%, or 75% of its final value

This makes the main phenomenon harder to attack:
persistent contender -> persistent winner -> persistent decisive margin.

The main paper uses only the exact-option readout with identity ordering. Controls are not used to support the main claim.
