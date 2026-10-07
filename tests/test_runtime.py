import re
import unittest

from hearthwork import runtime
from hearthwork.onboard import wanted_assets

# Asset names of the llama.cpp b11462 release.
ASSETS = """cudart-llama-b11462-bin-ubuntu-cuda-13.4-x64.tar.gz cudart-llama-bin-win-cuda-12.4-x64.zip
cudart-llama-bin-win-cuda-13.4-x64.zip llama-b11462-bin-macos-arm64.tar.gz llama-b11462-bin-ubuntu-arm64.tar.gz
llama-b11462-bin-ubuntu-cuda-12.8-x64.tar.gz llama-b11462-bin-ubuntu-cuda-13.4-x64.tar.gz
llama-b11462-bin-ubuntu-rocm-10.0-x64.tar.gz llama-b11462-bin-ubuntu-sycl-fp16-x64.tar.gz
llama-b11462-bin-ubuntu-sycl-fp32-x64.tar.gz llama-b11462-bin-ubuntu-vulkan-x64.tar.gz llama-b11462-bin-ubuntu-x64.tar.gz
llama-b11462-bin-win-cpu-x64.zip llama-b11462-bin-win-cuda-12.4-x64.zip llama-b11462-bin-win-cuda-13.4-x64.zip
llama-b11462-bin-win-rocm-10.0-x64.zip llama-b11462-bin-win-sycl-x64.zip llama-b11462-bin-win-vulkan-x64.zip
llama-b11462-bin-win-openvino-2026.4.1-x64.zip llama-b11462-bin-win-vulkan-arm64.zip""".split()


def matches(patterns):
    return [a for p in patterns for a in ASSETS if re.match(p, a)]


def hw(**kw):
    return {"os": "windows", "arch": "x64", "backend": "cpu", "gpus": [], **kw}


class AssetMapping(unittest.TestCase):
    def test_windows_x64(self):
        expected = {"cuda13": "llama-b11462-bin-win-cuda-13.4-x64.zip", "cuda12": "llama-b11462-bin-win-cuda-12.4-x64.zip",
                    "rocm": "llama-b11462-bin-win-rocm-10.0-x64.zip", "vulkan": "llama-b11462-bin-win-vulkan-x64.zip",
                    "sycl": "llama-b11462-bin-win-sycl-x64.zip", "cpu": "llama-b11462-bin-win-cpu-x64.zip"}
        for name, asset in expected.items():
            main, _ = runtime.assets_for(name, "windows", "x64")
            self.assertEqual(matches(main), [asset], name)

    def test_cuda_runtime_libraries_follow_the_cuda_version(self):
        _, extra = runtime.assets_for("cuda13", "windows", "x64")
        self.assertEqual(matches(extra), ["cudart-llama-bin-win-cuda-13.4-x64.zip"])
        _, extra = runtime.assets_for("cuda12", "windows", "x64")
        self.assertEqual(matches(extra), ["cudart-llama-bin-win-cuda-12.4-x64.zip"])
        _, extra = runtime.assets_for("cuda13", "linux", "x64")
        self.assertEqual(matches(extra), ["cudart-llama-b11462-bin-ubuntu-cuda-13.4-x64.tar.gz"])

    def test_linux_and_mac(self):
        self.assertEqual(matches(runtime.assets_for("rocm", "linux", "x64")[0]), ["llama-b11462-bin-ubuntu-rocm-10.0-x64.tar.gz"])
        self.assertEqual(matches(runtime.assets_for("sycl", "linux", "x64")[0]), ["llama-b11462-bin-ubuntu-sycl-fp16-x64.tar.gz"])
        self.assertEqual(matches(runtime.assets_for("cpu", "linux", "x64")[0]), ["llama-b11462-bin-ubuntu-x64.tar.gz"])
        self.assertEqual(matches(runtime.assets_for("cpu", "linux", "arm64")[0]), ["llama-b11462-bin-ubuntu-arm64.tar.gz"])
        self.assertEqual(matches(runtime.assets_for("metal", "macos", "arm64")[0]), ["llama-b11462-bin-macos-arm64.tar.gz"])

    def test_unavailable_runtimes(self):
        self.assertIsNone(runtime.assets_for("rocm", "windows", "arm64"))
        self.assertIsNone(runtime.assets_for("cuda13", "macos", "arm64"))
        self.assertIsNone(runtime.assets_for("metal", "windows", "x64"))

    def test_every_listed_runtime_has_patterns(self):
        for (os_, arch), names in runtime.AVAILABLE.items():
            for name in names:
                self.assertIsNotNone(runtime.assets_for(name, os_, arch), (os_, arch, name))
                self.assertIn(name, runtime.LABELS)

    def test_wanted_assets_follows_the_driver(self):
        main, _ = wanted_assets(hw(backend="cuda", cudaDriver="13.4"))
        self.assertEqual(matches(main), ["llama-b11462-bin-win-cuda-13.4-x64.zip"])
        main, extra = wanted_assets(hw(backend="cuda", cudaDriver="12.8"))
        self.assertEqual(matches(main), ["llama-b11462-bin-win-cuda-12.4-x64.zip"])
        self.assertEqual(len(matches(extra)), 1)

    def test_find_release_skips_releases_without_the_asset(self):
        def release(tag, names):
            return {"tag_name": tag, "assets": [{"name": n, "browser_download_url": "u/" + n} for n in names]}
        releases = [release("b2", []), release("v1", ["llama-b1-bin-win-vulkan-x64.zip"]),
                    release("b1", ["llama-b1-bin-win-vulkan-x64.zip"])]
        self.assertEqual(runtime.find_release("vulkan", hw(), releases), (1, [("llama-b1-bin-win-vulkan-x64.zip", "u/llama-b1-bin-win-vulkan-x64.zip")]))
        self.assertIsNone(runtime.find_release("cpu", hw(), releases))


class Recommendation(unittest.TestCase):
    def test_nvidia_by_driver(self):
        self.assertEqual(runtime.recommend(hw(backend="cuda", cudaDriver="13.4"))[0], "cuda13")
        self.assertEqual(runtime.recommend(hw(backend="cuda", cudaDriver="12.8"))[0], "cuda12")

    def test_amd_supported_generations_get_rocm(self):
        for gpu in ("AMD Radeon RX 7800 XT", "AMD Radeon RX 6700 XT", "AMD Radeon RX 9070", "AMD Radeon 780M Graphics",
                    "AMD Radeon(TM) 8060S Graphics", "AMD Radeon PRO W7900"):
            self.assertEqual(runtime.recommend(hw(backend="vulkan", gpus=[gpu]))[0], "rocm", gpu)

    def test_amd_older_generations_get_vulkan_with_a_reason(self):
        for gpu in ("AMD Radeon RX 580 Series", "Radeon RX 5700 XT", "AMD Radeon RX Vega 56", "AMD Radeon(TM) Graphics"):
            name, why = runtime.recommend(hw(backend="vulkan", gpus=[gpu]))
            self.assertEqual(name, "vulkan", gpu)
            self.assertIn("not supported", why)

    def test_rocm_needs_an_asset_for_this_os(self):
        name, why = runtime.recommend(hw(arch="arm64", backend="vulkan", gpus=["AMD Radeon RX 7800 XT"]))
        self.assertEqual(name, "vulkan")
        self.assertIn("no official ROCm", why)

    def test_intel_apple_cpu(self):
        self.assertEqual(runtime.recommend(hw(backend="vulkan", gpus=["Intel(R) Arc(TM) A770 Graphics"]))[0], "sycl")
        self.assertEqual(runtime.recommend(hw(backend="vulkan", gpus=["Intel(R) UHD Graphics 770"]))[0], "vulkan")
        self.assertEqual(runtime.recommend(hw(os="macos", arch="arm64", backend="metal"))[0], "metal")
        self.assertEqual(runtime.recommend(hw())[0], "cpu")


class SmokeTest(unittest.TestCase):
    def test_required_flags_come_from_the_server_command(self):
        flags = runtime.required_flags()
        for flag in ("-m", "-c", "-fa", "--jinja", "--fit", "--fit-target", "--load-mode", "--slot-save-path", "-kvu",
                     "-ctk", "-ctv", "-np", "-b", "-ub", "--host", "--port"):
            self.assertIn(flag, flags)
        self.assertNotIn("on", flags)

    def test_missing_flags(self):
        help_text = "-b,    --batch-size N\n-kvu, --kv-unified\n--fit [on|off]\n--fit-target MiB\n"
        self.assertEqual(runtime.missing_flags(help_text, ["-b", "-kvu", "--fit", "--fit-target"]), [])
        # a flag must match whole: "--fit" is not satisfied by "--fit-target" alone, "-b" not by "-ub"
        self.assertEqual(runtime.missing_flags("--fit-target MiB\n-ub N\n", ["--fit", "-b"]), ["--fit", "-b"])
        self.assertEqual(runtime.missing_flags(help_text, ["--load-mode"]), ["--load-mode"])


class UpdateNotice(unittest.TestCase):
    def config(self, **extra):
        return {"runtime": {"selected": "vulkan", "installed": {"vulkan": [1]}, "active": {"vulkan": 1},
                            "legacy": {"vulkan-b1": __file__}, **extra}}

    def test_notice_and_daily_limit(self):
        calls = []
        fetch = lambda: calls.append(1) or 5  # noqa: E731
        import hearthwork.runtime as rt
        saved, rt.save_config = rt.save_config, lambda c: None
        try:
            config = self.config()
            self.assertEqual(rt.update_notice(config, now=1e6, fetch=fetch), "llama.cpp b1 -> b5 available: hearthwork runtime update")
            self.assertEqual(rt.update_notice(config, now=1e6 + 60, fetch=fetch), "llama.cpp b1 -> b5 available: hearthwork runtime update")
            self.assertEqual(len(calls), 1)
            rt.update_notice(config, now=1e6 + 90000, fetch=fetch)
            self.assertEqual(len(calls), 2)
        finally:
            rt.save_config = saved

    def test_offline_is_silent(self):
        import hearthwork.runtime as rt

        def offline():
            raise OSError("no network")
        saved, rt.save_config = rt.save_config, lambda c: None
        try:
            self.assertIsNone(rt.update_notice(self.config(), now=1e6, fetch=offline))
        finally:
            rt.save_config = saved


if __name__ == "__main__":
    unittest.main()
