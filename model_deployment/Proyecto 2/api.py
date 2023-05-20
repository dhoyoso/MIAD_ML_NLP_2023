#!/usr/bin/python
from flask import Flask
from flask_restx import Api, Resource, fields
from m09_model_deployment import predict


# Definición aplicación Flask
app = Flask(__name__)

api = Api(
    app, 
    version='1.0',
    title='Movie Genre Prediction API',
    description='Movie Genre Prediction API')

ns = api.namespace('predict', 
     description='Movie Genre Classifier')
   
parser = api.parser()

# Arugmentos del API
parser.add_argument(
    'year', 
    type=int, 
    required=True, 
    help="Movie's year", 
    location='args')

parser.add_argument(
    'title', 
    type=str, 
    required=True, 
    help="Movie's title", 
    location='args')

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help="Movie's plot", 
    location='args')

parser.add_argument(
    'rating', 
    type=float, 
    required=True, 
    help="Movie's rating", 
    location='args')

nested_fields = {
    'P_Action': fields.Float,
    'P_Adventure': fields.Float,
    'P_Animation': fields.Float,
    'P_Biography': fields.Float,
    'P_Comedy': fields.Float,
    'P_Crime': fields.Float,
    'P_Documentary': fields.Float,
    'P_Drama': fields.Float,
    'P_Family': fields.Float,
    'P_Fantasy': fields.Float,
    'P_Film-Noir': fields.Float,
    'P_History': fields.Float,
    'P_Horror': fields.Float,
    'P_Music': fields.Float,
    'P_Musical': fields.Float,
    'P_Mystery': fields.Float,
    'P_News': fields.Float,
    'P_Romance': fields.Float,
    'P_Sci-Fi': fields.Float,
    'P_Short': fields.Float,
    'P_Sport': fields.Float,
    'P_Thriller': fields.Float,
    'P_War': fields.Float,
    'P_Western': fields.Float,
    'PRED_CLASSES': fields.String
}

# Definición del tipo y estructura de los datos de respuesta
resource_fields = api.model('Resource', {
    'movie_genre_result': fields.Nested(nested_fields),
})

# Definición de los métodos del API
@ns.route('/')
class MovieGenreApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "movie_genre_result": predict(args['year'], args['title'], args['plot'], args['rating'])
        }, 200
    

# Inicia el servidor de Flask
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)