'''from django.shortcuts import render

# Create your views here.
'''

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class PredictView(APIView):
    def post(self, request):
        # Extract input values from request
        v1 = request.data.get('v1')
        v2 = request.data.get('v2')
        v3 = request.data.get('v3')
        v4 = request.data.get('v4')
        v5 = request.data.get('v5')
        v6 = request.data.get('v6')
        v7 = request.data.get('v7')
        
        # Check if any value is None
        if None in [v1, v2, v3, v4, v5,v6,v7]:
            return Response({'error': 'One or more values are missing'}, status=status.HTTP_400_BAD_REQUEST)

        # Convert values to float
        try:
            v1 = int(v1)
            v2 = float(v2)
            v3 = int(v3)
            v4 = int(v4)
            v5 = float(v5)
            v6 = float(v6)
            v7 = int(v7)
        except ValueError:
            return Response({'error': 'One or more values are not valid numbers'}, status=status.HTTP_400_BAD_REQUEST)

        # Load the trained model and preprocess data
        import pandas as pd

        data=pd.read_csv('C:\\Users\\Sandhiya P\\Desktop\\MLOPS\\stroke prediction 2.csv')
        
        data['bmi']=data['bmi'].fillna(data['bmi'].mean())

        l=LabelEncoder()
        data['gender']=l.fit_transform(data['gender'])
        data['smoking_status']=l.fit_transform(data['smoking_status'])
        
        x=data.iloc[:,:-1]
        y=data.iloc[:,-1:]


        # Train the model
        from sklearn.linear_model import LogisticRegression
        lr=LogisticRegression()
        lr.fit(x,y)

        # Make predictions
        out = lr.predict(np.array([v1, v2, v3, v4, v5,v6,v7]).reshape(1, -1))
        prediction = int(out[0])

        # Return the prediction
        return Response({'prediction': prediction}, status=status.HTTP_200_OK)