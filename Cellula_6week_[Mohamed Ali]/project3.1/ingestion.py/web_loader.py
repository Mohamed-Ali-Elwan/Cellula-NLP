import requests

from bs4 import BeautifulSoup
from langchain_core.documents import Document


class WebLoader:

    @staticmethod
    def load_url(url: str) -> list[Document]:

        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        # Remove unnecessary HTML
        for tag in soup(
            ["script", "style", "noscript"]
        ):
            tag.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True,
        )

        return [
            Document(
                page_content=text,
                metadata={
                    "source": url,
                    "type": "url",
                },
            )
        ]

    @staticmethod
    def load_wikipedia(
        topic: str
    ) -> list[Document]:

        import wikipedia

        page = wikipedia.page(
            topic,
            auto_suggest=True,
        )

        return [
            Document(
                page_content=page.content,
                metadata={
                    "source": page.url,
                    "title": page.title,
                    "type": "wikipedia",
                },
            )
        ]