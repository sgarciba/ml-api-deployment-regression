from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_pascal


class PredictionResponse(BaseModel):

    predicted_price_usd: float
    model_version: str
    
    
class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_pascal,
        populate_by_name=True,
    )

    med_inc: float
    house_age: float
    ave_rooms: float 
    ave_bedrms: float
    population: float 
    ave_occup: float
    latitude: float
    longitude: float