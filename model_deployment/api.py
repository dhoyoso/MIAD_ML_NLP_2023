#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
from m09_model_deployment import predict


# Definición aplicación Flask
app = Flask(__name__)

api = Api(
    app, 
    version='1.0',
    title='Car Price Prediction API',
    description='Car Price Prediction API')

ns = api.namespace('predict', 
     description='Car Price Regressor')
   
parser = api.parser()

parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Year of the car', 
    location='args')

parser.add_argument(
    'Mileage', 
    type=int, 
    required=True, 
    help='Mileage of the car', 
    location='args')

parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='State in which the car was registered', 
    location='args')

parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='Make of the car', 
    location='args')

parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Model of the car', 
    location='args')

resource_fields = api.model('Resource', {
    'car_price_result': fields.Float,
})

@ns.route('/')
class CarPriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "car_price_result": predict(args['Year'], args['Mileage'], args['State'], args['Make'], args['Model'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
