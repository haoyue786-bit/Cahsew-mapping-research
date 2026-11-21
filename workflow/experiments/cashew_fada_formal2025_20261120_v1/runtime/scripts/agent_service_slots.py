"""Compatibility entry point for dispatching an Agent call for one training group."""


def call_in_slot(callback):
    """Run directly. The caller holds the group's own lease."""
    return callback()
