import unittest

from hearthwork.harnesses import normalize_anthropic


def request(header, first="hi"):
    return {
        "system": [{"type": "text", "text": f"x-anthropic-billing-header: cc_version=2.1.1.{header}; cc_entrypoint=sdk-cli;"},
                   {"type": "text", "text": "You are an agent."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": first}]},
            {"role": "system", "content": [{"type": "text", "text": "# Environment\nToday's date is 2026-01-01."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "next"},
            {"role": "system", "content": [{"type": "text", "text": "<total_tokens>14999854 tokens left</total_tokens>"}]},
        ],
    }


class NormalizeAnthropic(unittest.TestCase):
    def test_billing_header_and_token_counter_do_not_reach_the_model(self):
        out = normalize_anthropic(request("8f7"))
        self.assertEqual([b["text"] for b in out["system"]], ["You are an agent."])
        self.assertNotIn("total_tokens", str(out["messages"]))

    def test_sessions_differing_only_in_volatile_parts_normalize_equal(self):
        self.assertEqual(normalize_anthropic(request("8f7")), normalize_anthropic(request("05d")))

    def test_environment_info_kept_in_place_and_roles_alternate(self):
        out = normalize_anthropic(request("8f7"))
        roles = [m["role"] for m in out["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertIn("Today's date", out["messages"][0]["content"][-1]["text"])

    def test_plain_string_system_is_cleaned(self):
        body = {"system": "x-anthropic-billing-header: a\nKeep me", "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(normalize_anthropic(body)["system"], "Keep me")


if __name__ == "__main__":
    unittest.main()


class ChatTests(unittest.TestCase):
    def test_late_system_moves_into_user(self):
        from hearthwork.harnesses import normalize_chat
        body = {"model": "m", "messages": [
            {"role": "system", "content": "a"}, {"role": "developer", "content": "b"},
            {"role": "user", "content": "hi"}, {"role": "system", "content": "late"},
            {"role": "assistant", "content": "ok"}, {"role": "system", "content": "later"},
            {"role": "user", "content": "next"}]}
        out = normalize_chat(body)["messages"]
        self.assertEqual([m["role"] for m in out], ["system", "user", "assistant", "user"])
        self.assertEqual(out[0]["content"], "a\n\nb")
        self.assertIn("late", out[1]["content"])
        self.assertTrue(out[3]["content"].startswith("<system-reminder>\nlater"))

    def test_no_system_unchanged(self):
        from hearthwork.harnesses import normalize_chat
        body = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertIs(normalize_chat(body), body)
