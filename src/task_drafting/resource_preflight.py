"""Check configured GPU leases before drafting, without reserving runtime capacity."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.investigation.container_gpus import (
    GPU_ALLOCATION_BUSY_CATEGORY,
    GPU_ALLOCATION_INVALID_CATEGORY,
    GPU_ALLOCATION_LEASE_ROOT,
    GPU_ALLOCATION_QUARANTINED_CATEGORY,
    GpuLeaseError,
    GpuLeaseQuarantined,
    GpuLeaseTimeout,
    GpuSelectorError,
    _selector_devices,
    acquire_gpu_allocation_lease,
)
from src.investigation.gpu_quarantine import GpuQuarantine

MAX_PREFLIGHT_DEVICES = 16
MAX_GPU_IDENTIFIER_CHARS = 160
_DIAGNOSTICS = {
    GPU_ALLOCATION_BUSY_CATEGORY: "GPU allocation is busy; bounded infrastructure retry is allowed",
    GPU_ALLOCATION_INVALID_CATEGORY: "GPU allocation or lease storage is invalid",
    GPU_ALLOCATION_QUARANTINED_CATEGORY: (
        "GPU allocation is quarantined; runtime shutdown remains unverified"
    ),
}


@dataclass(frozen=True)
class ResourcePreflightReport:
    status: Literal["passed", "skipped", "infrastructure_failure"]
    category: str | None = None
    quarantines: tuple[GpuQuarantine, ...] = ()

    @property
    def exit_code(self) -> int:
        return 4 if self.status == "infrastructure_failure" else 0

    def fields(self) -> dict[str, object]:
        """Return bounded host metadata, never raw selectors, paths, or exception text."""
        fields: dict[str, object] = {
            "schema_version": 1,
            "gate": "resource_preflight",
            "stage": "resource_selection",
            "status": self.status,
            "category": self.category,
            "exit_code": self.exit_code,
            "diagnostic": _DIAGNOSTICS.get(self.category or "", ""),
        }
        if self.quarantines:
            fields["quarantines"] = [record.fields() for record in self.quarantines]
        return fields


def check_gpu_allocation(
    raw_selector: str | None,
    *,
    lease_root: Path = GPU_ALLOCATION_LEASE_ROOT,
    timeout_seconds: float = 1.0,
) -> ResourcePreflightReport:
    """Briefly acquire every selected GPU; runtime must acquire its own lease later."""
    if not raw_selector or not raw_selector.strip():
        return ResourcePreflightReport("skipped")
    if (
        len(raw_selector) > 8192
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 30
    ):
        return ResourcePreflightReport("infrastructure_failure", GPU_ALLOCATION_INVALID_CATEGORY)
    try:
        devices = _selector_devices(raw_selector)
        gpu_count = len(devices)
        if gpu_count > MAX_PREFLIGHT_DEVICES or any(
            len(device) > MAX_GPU_IDENTIFIER_CHARS for device in devices
        ):
            raise GpuSelectorError("GPU allocation exceeds diagnostic bounds")
        with acquire_gpu_allocation_lease(
            raw_selector,
            gpu_count,
            lease_root=lease_root,
            deadline=time.monotonic() + timeout_seconds,
            recovery_deadline=time.monotonic() + max(30.0, timeout_seconds),
        ):
            pass
    except GpuLeaseTimeout:
        return ResourcePreflightReport("infrastructure_failure", GPU_ALLOCATION_BUSY_CATEGORY)
    except GpuLeaseQuarantined as error:
        return ResourcePreflightReport(
            "infrastructure_failure", GPU_ALLOCATION_QUARANTINED_CATEGORY, error.quarantines,
        )
    except (GpuLeaseError, GpuSelectorError):
        return ResourcePreflightReport("infrastructure_failure", GPU_ALLOCATION_INVALID_CATEGORY)
    return ResourcePreflightReport("passed")


def _print_quarantine_diagnostics(report: ResourcePreflightReport) -> None:
    for record in report.quarantines:
        origin = record.origin or {}
        host = origin.get("hostname") or "unknown"
        marked = datetime.fromtimestamp(record.modified_at_ns / 1e9, timezone.utc).isoformat()
        print(
            f"GPU quarantine: {record.device}; marked {marked}; originating host {host} "
            f"({record.format}).", file=sys.stderr,
        )
        print(
            "Inspect on the same node: python -m src.investigation.gpu_lease_admin status "
            f"--device {record.device}", file=sys.stderr,
        )
    print(
        "Automatic recovery requires a complete owned-runtime record and verified shutdown. "
        "Legacy markers require explicit recovery after verifying evaluator/candidate containers "
        "and launch processes have stopped; do not remove lock files.",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    report = check_gpu_allocation(
        os.environ.get("AGENT_CONTAINER_GPUS"),
        lease_root=GPU_ALLOCATION_LEASE_ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report.fields(), sort_keys=True))
    if report.category == GPU_ALLOCATION_QUARANTINED_CATEGORY:
        _print_quarantine_diagnostics(report)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
