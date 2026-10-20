"""detect(): an AMD machine with a stray nvidia-smi prints warnings, not CSV; that must not crash."""
import unittest
from unittest import mock

from hearthwork import onboard

WDDM = "No devices were found, and NVIDIA GPU is in WDDM mode.\n"  # the real user's output
RTX = "NVIDIA GeForce RTX 5060 Ti, 16384\n"
AMD_NAME = "AMD Radeon RX 7900 XTX\n"
AMD_VRAM = "AMD Radeon RX 7900 XTX|25769803776\n"  # 24 GiB in bytes


def detect(nvidia_rows, banner="", driver=""):
    """Run detect() with every command answered from canned output (nvidia-smi always on PATH)."""
    def fake(command):
        joined = " ".join(command)
        if command[0] != "nvidia-smi":
            if "Win32_VideoController" in joined:
                return AMD_NAME
            if "qwMemorySize" in joined:
                return AMD_VRAM
            return ""
        if "name,memory.total" in joined:
            return nvidia_rows
        if "driver_version" in joined:
            return driver
        return banner
    with mock.patch.object(onboard, "run", fake), \
         mock.patch.object(onboard.shutil, "which", lambda e: r"C:\fake\nvidia-smi.exe" if e == "nvidia-smi" else None), \
         mock.patch.object(onboard, "memory_gb", lambda: (32.0, 24.0)), \
         mock.patch.object(onboard, "WINDOWS", True), \
         mock.patch.object(onboard, "MAC", False):
        return onboard.detect()


class NvidiaSmiWarnings(unittest.TestCase):
    def test_warning_line_falls_through_to_vulkan(self):
        hw = detect(WDDM)
        self.assertEqual(hw["backend"], "vulkan")  # no NVIDIA GPU found, so the AMD path runs
        self.assertEqual(hw["gpus"], ["AMD Radeon RX 7900 XTX"])
        self.assertEqual(hw["vramGB"], 24.0)
        self.assertNotIn("cudaDriver", hw)

    def test_real_row_parses(self):
        self.assertEqual(onboard.nvidia_gpu_rows(RTX.splitlines()), [("NVIDIA GeForce RTX 5060 Ti", 16.0)])
        hw = detect(RTX, banner="| NVIDIA-SMI 570 | CUDA Version: 12.8")
        self.assertEqual(hw["backend"], "cuda")
        self.assertEqual(hw["gpus"], ["NVIDIA GeForce RTX 5060 Ti (16 GB)"])
        self.assertEqual(hw["vramGB"], 16.0)
        self.assertEqual(hw["cudaDriver"], "12.8")

    def test_mixed_warning_and_row_counts_only_the_gpu(self):
        hw = detect(WDDM + RTX, banner="| NVIDIA-SMI 570 | CUDA Version: 12.8")
        self.assertEqual(hw["backend"], "cuda")
        self.assertEqual(hw["gpus"], ["NVIDIA GeForce RTX 5060 Ti (16 GB)"])
        self.assertEqual(hw["vramGB"], 16.0)

    def test_driver_query_warning_falls_back(self):  # no CUDA version in the banner -> driver_version query
        self.assertEqual(detect(RTX, driver=WDDM)["cudaDriver"], "12.0")
        self.assertEqual(detect(RTX, driver="580.65.06\n")["cudaDriver"], "13.0")


if __name__ == "__main__":
    unittest.main()
