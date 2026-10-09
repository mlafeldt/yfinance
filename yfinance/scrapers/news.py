from yfinance import utils
from yfinance.config import YfConfig
from yfinance.const import _CONTENT_GATEWAY_URL_
from yfinance.data import YfData
from yfinance.exceptions import YFDataException


_TAB_LISTS = {
    "all": "finance-US-en-US-ticker-all",
    "news": "finance-US-en-US-ticker-news",
    "press releases": "finance-US-en-US-ticker-pressrelease",
}

# Larger pages make the gateway fail to resolve many assets (AssetError).
_MAX_PAGE_SIZE = 50

# Ticker news query for Yahoo's content gateway, as used by the quote page.
# Aliases match the field names of the retired /xhr/ncp endpoint. Keep
# clickthroughUrl: without a field using $clientContext, every asset comes back
# as AssetError.
_QUERY = """
query($listInput: LightyearListInput!, $clientContext: ClientContext!, $mlRecsInput: MLRecsInput!, $first: Int, $after: String) {
  lightyearList: financeTickerList(list_input: $listInput, cc: $clientContext) {
    main: mlRecsStream(mlRecsInput: $mlRecsInput, first: $first, after: $after) {
      pageInfo { hasNextPage endCursor }
      edges { node { ... on AssetRef { asset {
        __typename
        ... on Story { id: uuid title contentAttributes { ...Attrs } finance { stockTickers { symbol } } }
        ... on Video { id: uuid title contentAttributes { ...Attrs } finance { stockTickers { symbol } } }
      } } } }
    }
  }
}
fragment Attrs on ContentAttributes {
  description summary pubDate: publishTime displayTime isHosted canonicalUrl
  clickThroughUrl: clickthroughUrl(cc: $clientContext)
  provider { displayName url sourceId: providerId }
  thumbnail {
    originalUrl: url originalWidth: width originalHeight: height
    resolutions: resized(resizeInput: [{operations: ["resizefill_w170_h128"], transformLabel: "170x128"}]) {
      url width height tag: transformLabel
    }
  }
}
"""


def fetch_news(data: YfData, symbol: str, count: int, tab: str) -> list:
    """Fetch up to `count` news items, paging past unresolvable assets."""
    list_alias = _TAB_LISTS.get(tab.lower())
    if not list_alias:
        raise ValueError(f"Invalid tab name '{tab}'. Choose from: {', '.join(_TAB_LISTS)}")

    variables = {
        "listInput": {
            "slug": f"list={list_alias}",
            "getFullList": True,
            "queryVariables": {"tickerSymbol": [symbol]},
        },
        "clientContext": {"lang": "en-US", "region": "US", "site": "finance", "device": "desktop"},
        "mlRecsInput": {
            "instance": "FINANCE",
            "count": 200,
            "rankInput": {"financeQuoteNewsV2": {"tickerSymbol": [symbol]}},
        },
    }

    news = []
    after = None
    while len(news) < count:
        try:
            page = _fetch_page(data, {**variables, "first": min(count - len(news), _MAX_PAGE_SIZE), "after": after})
        except (YFDataException, ValueError) as e:
            if not YfConfig.debug.hide_exceptions:
                raise YFDataException(f"{symbol}: Failed to retrieve news: {e}") from e
            utils.get_yf_logger().error(f"{symbol}: Failed to retrieve news: {e}")
            break

        assets = [(edge.get("node") or {}).get("asset") for edge in page.get("edges", [])]
        news += [item for item in map(_parse_asset, assets) if item]

        page_info = page.get("pageInfo") or {}
        after = page_info.get("endCursor")
        if not page_info.get("hasNextPage") or not after:
            break

    # Enforce the public limit even if the gateway returns more than requested.
    return news[:count]


def _fetch_page(data: YfData, variables: dict) -> dict:
    response = data.post(_CONTENT_GATEWAY_URL_, body={"query": _QUERY, "variables": variables},
                         headers={"x-yahoo-cg-client-name": "finance"})
    if response.status_code != 200:
        raise YFDataException(f"HTTP {response.status_code}: {response.text[:200]}")
    page = ((response.json().get("data") or {}).get("lightyearList") or {}).get("main")
    if page is None:
        raise YFDataException(f"unexpected response: {response.text[:200]}")
    return page


def _parse_asset(asset):
    # The query aliases fields to their legacy names; only override what
    # GraphQL can't produce in the format of the retired /xhr/ncp endpoint.
    if not asset or asset.get("__typename") not in ("Story", "Video"):
        return None
    attrs = asset.get("contentAttributes") or {}
    clickthrough_url = attrs.get("clickThroughUrl")
    url = attrs.get("canonicalUrl") or clickthrough_url
    if not url:
        return None
    return {
        "id": asset["id"],
        "content": {
            "id": asset["id"],
            "contentType": asset["__typename"].upper(),
            "title": asset.get("title"),
            **attrs,
            "finance": asset.get("finance"),
            "canonicalUrl": {"url": url},
            "clickThroughUrl": {"url": clickthrough_url} if clickthrough_url else None,
        },
    }
