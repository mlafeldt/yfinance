"""Tests for yfinance.data internal helpers."""
import unittest
from functools import lru_cache
from unittest import mock

from yfinance._http import new_session
from yfinance.config import YfConfig
from yfinance.data import SingletonMeta, YfData, _normalize_proxy, lru_cache_freezeargs
from yfinance.exceptions import YFDataException, YFRateLimitError
from yfinance.utils import frozendict

_ISIN_LOOKUP_URL = "https://markets.businessinsider.com/ajax/SearchController_Suggest"


class TestProxyConfig(unittest.TestCase):
    def test_proxy_string_applies_to_http_and_https(self):
        proxy = "http://user:pass@example.com:8080"

        self.assertEqual(
            _normalize_proxy(proxy),
            {"http": proxy, "https": proxy},
        )

    def test_proxy_mapping_is_preserved(self):
        proxy = {"http": "http://proxy:8080", "https": "https://proxy:8443"}

        self.assertIs(_normalize_proxy(proxy), proxy)

    def test_proxy_none_is_preserved(self):
        self.assertIsNone(_normalize_proxy(None))


class TestFrozenDict(unittest.TestCase):
    def test_is_dict_subclass(self):
        d = frozendict({"a": 1, "b": 2})
        self.assertEqual(d["a"], 1)
        self.assertEqual(d.get("b"), 2)
        self.assertEqual(set(d.keys()), {"a", "b"})
        self.assertIsInstance(d, dict)

    def test_is_hashable(self):
        d1 = frozendict({"a": 1, "b": 2})
        d2 = frozendict({"b": 2, "a": 1})
        self.assertEqual(hash(d1), hash(d2))
        self.assertEqual(len({d1, d2}), 1)

    def test_mutation_raises(self):
        d = frozendict({"a": 1})
        with self.assertRaises(TypeError):
            d["b"] = 2
        with self.assertRaises(TypeError):
            del d["a"]
        with self.assertRaises(AttributeError):
            d.pop("a")
        with self.assertRaises(AttributeError):
            d.clear()
        with self.assertRaises(AttributeError):
            d.update({"x": 9})

    def test_lru_cache_integration(self):
        call_count = {"n": 0}

        @lru_cache_freezeargs
        @lru_cache(maxsize=8)
        def fn(params):
            call_count["n"] += 1
            return sum(params.values())

        self.assertEqual(fn({"a": 1, "b": 2}), 3)
        self.assertEqual(fn({"a": 1, "b": 2}), 3)
        self.assertEqual(call_count["n"], 1)
        self.assertEqual(fn({"a": 1, "b": 3}), 4)
        self.assertEqual(call_count["n"], 2)


class TestYfDataLoginCookies(unittest.TestCase):
    def test_set_login_cookies_invalidates_crumb(self):
        # Logging in (or switching accounts) must drop a crumb minted under a
        # different login state so the next request re-mints a matched crumb.
        # Use a throwaway instance (YfData is a singleton) to avoid leaking the
        # test cookies into the shared session other tests rely on.
        SingletonMeta._instances.pop(YfData, None)
        try:
            data = YfData()
            data._crumb = "stale-crumb-sentinel"
            data.set_login_cookies("T-test-value", "Y-test-value")
            self.assertIsNone(data._crumb)
        finally:
            SingletonMeta._instances.pop(YfData, None)


class TestCookieStrategyLoginPreservation(unittest.TestCase):
    """The csrf->basic strategy toggle must not wipe user-set login cookies.

    Uses a throwaway YfData (the class is a SingletonMeta singleton) so the test
    cookies / login flag don't leak into the shared session other tests use.
    """

    def setUp(self):
        SingletonMeta._instances.pop(YfData, None)
        self.data = YfData()
        self.data._cookie_strategy = 'csrf'
        self.data._cookie = None
        self.data._crumb = None
        self.data._logged_in = False
        self.data._session.cookies.clear()
        self.data._session.cookies.update({"T": "tval", "Y": "yval", "A3": "a3val"})

    def tearDown(self):
        SingletonMeta._instances.pop(YfData, None)

    def test_logged_in_switch_keeps_login_cookies(self):
        self.data._logged_in = True
        self.data._set_cookie_strategy('basic')  # csrf->basic == the clearing branch
        self.assertEqual(self.data._session.cookies.get("T"), "tval")
        self.assertEqual(self.data._session.cookies.get("Y"), "yval")

    def test_logged_out_switch_still_clears_jar(self):
        # Not logged in: the toggle clears as before, so a bad/anonymous session
        # can recover a fresh anonymous cookie.
        self.data._logged_in = False
        self.data._set_cookie_strategy('basic')
        self.assertIsNone(self.data._session.cookies.get("T"))
        self.assertIsNone(self.data._session.cookies.get("Y"))

    def test_switch_resets_crumb_even_when_logged_in(self):
        # The anonymous cookie/crumb refresh path must stay intact.
        self.data._logged_in = True
        self.data._crumb = "stale-crumb"
        self.data._set_cookie_strategy('basic')
        self.assertIsNone(self.data._crumb)

    def test_set_login_cookies_marks_logged_in(self):
        self.data._logged_in = False
        self.data.set_login_cookies("T-v", "Y-v")
        self.assertTrue(self.data._logged_in)

    def test_optimistic_login_survives_strategy_toggle(self):
        # set_login_cookies marks _logged_in True optimistically; a strategy
        # toggle that fires before check_login confirms (e.g. the subscriptions
        # request itself 4xx-ing) must still keep the freshly-set T/Y.
        self.data._session.cookies.clear()
        self.data.set_login_cookies("T-opt", "Y-opt")  # -> _logged_in = True
        self.data._cookie_strategy = 'csrf'
        self.data._set_cookie_strategy('basic')
        self.assertEqual(self.data._session.cookies.get("T"), "T-opt")
        self.assertEqual(self.data._session.cookies.get("Y"), "Y-opt")


class TestSetSessionCacheDetection(unittest.TestCase):
    """A session is caching only when a cache is active, not when `.cache` merely exists.

    curl_cffi >= 0.16 Sessions always expose `.cache` (None when disabled).
    """

    def setUp(self):
        SingletonMeta._instances.pop(YfData, None)
        self.data = YfData()

    def tearDown(self):
        SingletonMeta._instances.pop(YfData, None)

    def test_session_with_disabled_cache_is_accepted(self):
        session = new_session()
        session.cache = None
        self.data._set_session(session)
        self.assertIs(self.data._session, session)

    def test_session_with_active_cache_is_rejected(self):
        session = new_session()
        session.cache = object()
        with self.assertRaises(YFDataException):
            self.data._set_session(session)


class TestRequestAuthentication(unittest.TestCase):
    def setUp(self):
        singleton = mock.patch.dict(SingletonMeta._instances, clear=True)
        singleton.start()
        self.addCleanup(singleton.stop)
        self.data = YfData()
        self.addCleanup(self.data._session.close)

    def test_non_finance_4xx_preserves_finance_authentication(self):
        for status in (400, 401):
            with self.subTest(status=status):
                self.data._cookie_strategy = "csrf"
                self.data._cookie = True
                self.data._crumb = "finance-crumb"
                self.data._session.cookies.update({"A3": "finance-cookie"})
                response = mock.MagicMock(status_code=status, url=_ISIN_LOOKUP_URL)
                with mock.patch.object(self.data, "_get_cookie_and_crumb", return_value=("finance-crumb", "csrf")) as auth, \
                     mock.patch.object(self.data._session, "get", return_value=response) as get:
                    result = self.data.get(_ISIN_LOOKUP_URL, params={"query": "AAPL"})

                self.assertEqual(self.data._cookie_strategy, "csrf")
                self.assertTrue(self.data._cookie)
                self.assertEqual(self.data._crumb, "finance-crumb")
                self.assertEqual(self.data._session.cookies.get("A3"), "finance-cookie")
                self.assertIs(result, response)
                auth.assert_not_called()
                get.assert_called_once()
                self.assertNotIn("crumb", get.call_args.kwargs["params"])

    def test_non_finance_get_skips_crumb(self):
        urls = (_ISIN_LOOKUP_URL,
                "https://finance.yahoo.com.example.com/")
        for url in urls:
            with self.subTest(url=url):
                response = mock.MagicMock(status_code=200, url=url)
                params = {"query": "AAPL"}
                with mock.patch.object(self.data, "_get_cookie_and_crumb", return_value=("finance-crumb", "basic")) as auth, \
                     mock.patch.object(self.data._session, "get", return_value=response) as get:
                    self.assertIs(self.data.get(url, params=params, timeout=7), response)
                auth.assert_not_called()
                get.assert_called_once_with(url=url, params={"query": "AAPL"}, timeout=7)
                self.assertEqual(params, {"query": "AAPL"})

    def test_finance_4xx_refreshes_crumb(self):
        urls = ("https://finance.yahoo.com/",
                "https://query1.finance.yahoo.com/v7/finance/quote",
                "https://query2.finance.yahoo.com/v10/finance/quoteSummary/AAPL")
        for url in urls:
            with self.subTest(url=url):
                self.data._cookie_strategy = "csrf"
                response = mock.MagicMock(status_code=200)
                params = {"symbol": "AAPL"}
                with mock.patch.object(self.data, "_get_cookie_and_crumb",
                                       side_effect=[("old-crumb", "csrf"), ("new-crumb", "basic")]) as auth, \
                     mock.patch.object(self.data._session, "post",
                                       side_effect=[mock.MagicMock(status_code=401), response]) as post:
                    self.assertIs(self.data.post(url, params=params, timeout=7), response)
                self.assertEqual(post.call_count, 2)
                self.assertEqual(self.data._cookie_strategy, "basic")
                auth.assert_has_calls([mock.call(30), mock.call(7)])
                self.assertEqual(post.call_args.kwargs["params"]["crumb"], "new-crumb")
                self.assertEqual(params, {"symbol": "AAPL"})

    def test_non_finance_request_preserves_options(self):
        response = mock.MagicMock(status_code=200)
        body = {"query": "AAPL"}
        params = {"example": "value"}
        with mock.patch.dict(YfConfig.network.data, {"proxy": "http://proxy.example.com:8080"}), \
             mock.patch.object(self.data, "_get_cookie_and_crumb") as auth, \
             mock.patch.object(self.data._session, "post", return_value=response) as post:
            self.assertIs(self.data.post(_ISIN_LOOKUP_URL, body=body, params=params, timeout=7), response)
        auth.assert_not_called()
        post.assert_called_once_with(url=_ISIN_LOOKUP_URL, json=body, params=params, timeout=7)
        self.assertEqual(self.data._session.proxies,
                         {"http": "http://proxy.example.com:8080", "https": "http://proxy.example.com:8080"})
        self.assertEqual(params, {"example": "value"})

    def test_non_finance_transport_error_retries(self):
        response = mock.MagicMock(status_code=200, url=_ISIN_LOOKUP_URL)
        with mock.patch.dict(YfConfig.network.data, {"retries": 1}), \
             mock.patch("yfinance.data._time.sleep") as sleep, \
             mock.patch.object(self.data, "_get_cookie_and_crumb") as auth, \
             mock.patch.object(self.data._session, "get", side_effect=[TimeoutError("timed out"), response]) as get:
            self.assertIs(self.data.get(_ISIN_LOOKUP_URL), response)
        auth.assert_not_called()
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_non_finance_429_raises_without_cookie_retry(self):
        response = mock.MagicMock(status_code=429, url=_ISIN_LOOKUP_URL)
        with mock.patch.object(self.data, "_get_cookie_and_crumb") as auth, \
             mock.patch.object(self.data._session, "get", return_value=response) as get:
            with self.assertRaises(YFRateLimitError):
                self.data.get(_ISIN_LOOKUP_URL)
        auth.assert_not_called()
        get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
