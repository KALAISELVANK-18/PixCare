class MapModel {
  MapModel({
    required this.placeId,
    required this.name,
    required this.vicinity,
    required this.latitude,
    required this.longitude,
    required this.userRatingsTotal,
  });

  late final String placeId; // Change place_id to String
  late final String name;
  late final String vicinity;
  late final double latitude;
  late final double longitude;
  late final int userRatingsTotal;

  MapModel.fromJson(Map<String, dynamic> json) {
    placeId = json['place_id'];
    name = json['name'];
    vicinity = json['vicinity'];
    latitude = json['geometry']['location']['lat'];
    longitude = json['geometry']['location']['lng'];
    final userRatingsTotalValue = json['user_ratings_total'];
    if (userRatingsTotalValue is int) {
      userRatingsTotal = userRatingsTotalValue;
    } else {
      // Handle the case where 'user_ratings_total' is missing or not an int
      // You can assign a default value or handle it according to your logic.
      userRatingsTotal = 0; // Default value
    } // Parse as int
  }

  Map<String, dynamic> toJson() {
    final _data = <String, dynamic>{};
    _data['place_id'] = placeId;
    _data['name'] = name;
    _data['vicinity'] = vicinity;
    _data['latitude'] = latitude;
    _data['longitude'] = longitude;
    _data['user_ratings_total'] = userRatingsTotal;

    return _data;
  }
}
