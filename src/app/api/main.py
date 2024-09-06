from typing import Annotated

from fastapi import FastAPI, Query, status

from src.app.models.ResponseModels import ForecastData


class APIManager:

    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        self.app.add_api_route("/", self.read_root, methods=["GET"], status_code=status.HTTP_200_OK)
        self.app.add_api_route("/help", self.get_help, methods=["GET"], status_code=status.HTTP_200_OK)
        self.app.add_api_route("/predict", self.predict, methods=["GET"], status_code=status.HTTP_200_OK,
                               response_model=ForecastData)

    async def read_root(self):
        return {"message": "Welcome to SIH Backend!"}

    async def get_help(self):
        return {
            "name": "SIH Backend",
            "version": 1.0
        }

    async def predict(
            self,
            month: Annotated[str, Query(...)]
    ):
        pass
