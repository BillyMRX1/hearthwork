import unittest
from pathlib import Path

from hearthwork import api


class Headers(dict):
    def get(self, key, default=None):
        return super().get(key.lower(), default)


MODELS = [Path("a/Qwen3.5-35B-A3B-Q4.gguf"), Path("a/Qwen3-Coder-30B.gguf"), Path("a/Llama-3.1-8B.gguf")]


class AuthTests(unittest.TestCase):
    def test_no_key_required(self):
        self.assertTrue(api.key_ok(None, Headers()))

    def test_bearer_and_x_api_key(self):
        self.assertTrue(api.key_ok("s3", Headers({"authorization": "Bearer s3"})))
        self.assertTrue(api.key_ok("s3", Headers({"x-api-key": "s3"})))

    def test_wrong_or_missing(self):
        self.assertFalse(api.key_ok("s3", Headers()))
        self.assertFalse(api.key_ok("s3", Headers({"authorization": "Bearer nope"})))
        self.assertFalse(api.key_ok("s3", Headers({"x-api-key": ""})))


class MatchTests(unittest.TestCase):
    cur = "Qwen3.5-35B-A3B-Q4"

    def test_unknown_name_uses_current(self):
        self.assertEqual(api.match_model(MODELS, "claude-sonnet-4", self.cur), ("current", None))
        self.assertEqual(api.match_model(MODELS, "", self.cur), ("current", None))
        self.assertEqual(api.match_model(MODELS, None, None), ("current", None))

    def test_current_model_no_switch(self):
        self.assertEqual(api.match_model(MODELS, "QWEN3.5", self.cur), ("current", None))

    def test_switch_case_insensitive_substring(self):
        self.assertEqual(api.match_model(MODELS, "llama", self.cur), ("switch", MODELS[2]))
        self.assertEqual(api.match_model(MODELS, "coder", self.cur), ("switch", MODELS[1]))

    def test_exact_wins_and_ambiguous(self):
        self.assertEqual(api.match_model(MODELS, "qwen3-coder-30b", self.cur), ("switch", MODELS[1]))
        self.assertEqual(api.match_model(MODELS, "q", "Llama-3.1-8B")[0], "ambiguous")


class ListingTests(unittest.TestCase):
    def test_running_first_no_duplicates(self):
        data = api.models_listing(MODELS, "Llama-3.1-8B")["data"]
        self.assertEqual([d["id"] for d in data], ["Llama-3.1-8B", "Qwen3.5-35B-A3B-Q4", "Qwen3-Coder-30B"])

    def test_none_running(self):
        self.assertEqual(len(api.models_listing(MODELS, None)["data"]), 3)


class FakeHandler:
    def __init__(self, command, path, body=b"", headers=None):
        import io
        self.command, self.path, self.headers = command, path, Headers(headers or {})
        self.rfile, self.sent = io.BytesIO(body), None
        self.headers["content-length"] = str(len(body))

    def read_body(self, limit=None):
        return self.rfile.read(int(self.headers["content-length"]))

    def _send_json(self, status, payload):
        self.sent = (status, payload)


class SwitchTests(unittest.TestCase):
    def make(self, allow, current="Qwen3.5-35B-A3B-Q4"):
        a = api.Api({"server": {"port": 1}, "modelsDir": "x"}, 8080, None, allow)
        a.models = lambda: MODELS
        self.switched = []
        a.switch_to = lambda m: self.switched.append(m) or True
        for name, fake in (("served_model", lambda port, timeout=2: current), ("port_open", lambda port: True)):
            self.addCleanup(setattr, api, name, getattr(api, name))
            setattr(api, name, fake)
        return a

    def post(self, a, model):
        h = FakeHandler("POST", "/v1/chat/completions", ('{"model": "%s"}' % model).encode())
        return a.intercept(h), h

    def test_switch_when_allowed_and_body_kept(self):
        a = self.make(True)
        done, h = self.post(a, "llama")
        self.assertFalse(done)
        self.assertEqual(self.switched, [MODELS[2]])
        self.assertEqual(h.rfile.read(), b'{"model": "llama"}')

    def test_no_switch_when_not_allowed(self):
        a = self.make(False)
        self.assertFalse(self.post(a, "llama")[0])
        self.assertEqual(self.switched, [])

    def test_ambiguous_is_400(self):
        a = self.make(True, current="Llama-3.1-8B")
        done, h = self.post(a, "q")
        self.assertTrue(done)
        self.assertEqual(h.sent[0], 400)

    def test_auth_401_and_other_paths_404(self):
        a = self.make(False)
        a.key = "k"
        h = FakeHandler("GET", "/v1/models")
        self.assertTrue(a.intercept(h))
        self.assertEqual(h.sent[0], 401)
        h = FakeHandler("GET", "/slots", headers={"x-api-key": "k"})
        a.intercept(h)
        self.assertEqual(h.sent[0], 404)

    def test_models_listing_served_locally(self):
        a = self.make(False)
        h = FakeHandler("GET", "/v1/models")
        self.assertTrue(a.intercept(h))
        self.assertEqual(h.sent[1]["data"][0]["id"], "Qwen3.5-35B-A3B-Q4")


if __name__ == "__main__":
    unittest.main()
