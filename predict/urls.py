from django.urls import path
from .views import predict_traffic, best_route, area_suggestions

urlpatterns = [
    path("predict/", predict_traffic),
    path("best-route/", best_route),
    path("area-suggestions/", area_suggestions),
]
