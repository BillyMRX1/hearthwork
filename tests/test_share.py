import json
import unittest

from hearthwork import share
from hearthwork.remote import parse_host


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.pairing = share.Pairing(self.clock)

    def start(self):
        request, pin = self.pairing.start("laptop", "192.168.0.9")
        return request, pin

    def wrong(self, pin):
        return "000000" if pin != "000000" else "111111"

    def test_pin_is_six_digits_and_right_pin_pairs_once(self):
        request, pin = self.start()
        self.assertRegex(pin, r"^\d{6}$")
        self.assertEqual(self.pairing.confirm(request, pin), ("ok", "laptop"))
        self.assertEqual(self.pairing.confirm(request, pin)[0], "unknown")

    def test_pin_expires_after_two_minutes(self):
        request, pin = self.start()
        self.clock.now += share.PIN_SECONDS - 1
        self.assertEqual(self.pairing.confirm(request, "x" * 6)[0], "wrong")
        request2, pin2 = self.start()
        self.clock.now += share.PIN_SECONDS + 1
        self.assertEqual(self.pairing.confirm(request2, pin2)[0], "expired")

    def test_three_wrong_pins_end_the_request_even_for_the_right_pin(self):
        request, pin = self.start()
        self.assertEqual(self.pairing.confirm(request, self.wrong(pin)), ("wrong", 2))
        self.assertEqual(self.pairing.confirm(request, self.wrong(pin)), ("wrong", 1))
        self.assertEqual(self.pairing.confirm(request, self.wrong(pin)), ("wrong", 0))
        self.assertEqual(self.pairing.confirm(request, pin)[0], "unknown")

    def test_five_failures_lock_pairing_for_ten_minutes(self):
        for _ in range(2):
            request, pin = self.start()
            for _ in range(3):
                status = self.pairing.confirm(request, self.wrong(pin))[0]
        self.assertEqual(status, "locked")
        self.assertIsNone(self.pairing.start("laptop", "1.2.3.4"))
        self.clock.now += share.LOCK_SECONDS - 1
        self.assertIsNone(self.pairing.start("laptop", "1.2.3.4"))
        self.clock.now += 2
        self.assertIsNotNone(self.pairing.start("laptop", "1.2.3.4"))

    def test_failures_older_than_the_window_do_not_count(self):
        request, pin = self.start()
        for _ in range(3):
            self.pairing.confirm(request, self.wrong(pin))
        self.clock.now += share.LOCK_WINDOW + 1
        request, pin = self.start()
        self.assertEqual(self.pairing.confirm(request, self.wrong(pin))[0], "wrong")
        self.assertFalse(self.pairing.locked())


class DeviceTests(unittest.TestCase):
    def test_only_the_hash_is_stored_and_the_key_matches(self):
        key = "secret-key"
        devices = [{"name": "a", "keyHash": share.hash_key("other")}, {"name": "b", "keyHash": share.hash_key(key)}]
        self.assertNotIn(key, json.dumps(devices))
        self.assertEqual(share.find_device(devices, key)["name"], "b")
        self.assertIsNone(share.find_device(devices, "wrong"))
        self.assertIsNone(share.find_device(devices, ""))
        self.assertIsNone(share.find_device([], key))

    def test_key_from_bearer_or_x_api_key(self):
        self.assertEqual(share.request_key({"authorization": "Bearer abc"}), "abc")
        self.assertEqual(share.request_key({"authorization": "bearer abc"}), "abc")
        self.assertEqual(share.request_key({"x-api-key": "xyz"}), "xyz")
        self.assertEqual(share.request_key({}), "")


class DiscoveryTests(unittest.TestCase):
    def test_reply_round_trip(self):
        data = share.discovery_reply("pc", 8484, "Qwen")
        self.assertEqual(share.parse_discovery(data, "192.168.0.7"),
                         {"host": "192.168.0.7", "port": 8484, "hostname": "pc", "model": "Qwen", "version": share.__version__,
                      "lan": True, "tailscaleIps": []})

    def test_garbage_is_ignored(self):
        for data in (b"", b"HEARTHWORK?", b"{}", b'{"hostname": "x"}', b"\xff\xfe"):
            self.assertIsNone(share.parse_discovery(data, "1.1.1.1"))

    def test_public_profiles(self):
        self.assertEqual(share.parse_profiles("Home|Private\nCafe WiFi|Public\nTailscale|Public\n"), ["Cafe WiFi", "Tailscale"])
        self.assertEqual(share.parse_profiles("Home|Private\ncorp|DomainAuthenticated"), [])

    def test_parse_host(self):
        self.assertEqual(parse_host("10.0.0.2"), ("10.0.0.2", share.DEFAULT_PORT))
        self.assertEqual(parse_host("10.0.0.2:9000"), ("10.0.0.2", 9000))
        with self.assertRaises(ValueError):
            parse_host("10.0.0.2:x")


if __name__ == "__main__":
    unittest.main()
