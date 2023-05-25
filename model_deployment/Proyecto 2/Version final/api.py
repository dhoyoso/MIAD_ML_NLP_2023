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

nested_fields = api.model('NestedFields', {
    'p_Action': fields.Float,
    'p_Adventure': fields.Float,
    'p_Animation': fields.Float,
    'p_Biography': fields.Float,
    'p_Comedy': fields.Float,
    'p_Crime': fields.Float,
    'p_Documentary': fields.Float,
    'p_Drama': fields.Float,
    'p_Family': fields.Float,
    'p_Fantasy': fields.Float,
    'p_Film-Noir': fields.Float,
    'p_History': fields.Float,
    'p_Horror': fields.Float,
    'p_Music': fields.Float,
    'p_Musical': fields.Float,
    'p_Mystery': fields.Float,
    'p_News': fields.Float,
    'p_Romance': fields.Float,
    'p_Sci-Fi': fields.Float,
    'p_Short': fields.Float,
    'p_Sport': fields.Float,
    'p_Thriller': fields.Float,
    'p_War': fields.Float,
    'p_Western': fields.Float,
    'PRED_CLASSES': fields.String
})

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
         "movie_genre_result": predict(args['year'], args['title'], args['plot'])
        }, 200
    

# Inicia el servidor de Flask
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)