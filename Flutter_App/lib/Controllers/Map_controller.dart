import 'dart:convert';
import 'dart:developer';

import 'package:get/get.dart';

import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:http/http.dart' as http;

import 'package:geocoding/geocoding.dart';
import 'package:geolocator/geolocator.dart';

import '../Models/Model.dart';

class MapController extends GetxController {
  List<MapModel> mapModel = <MapModel>[].obs;
  var markers = RxSet<Marker>();
  var isLoading = false.obs;

  fetchLocations(double latitude, double longitude) async {
    try {
      isLoading(true);
      http.Response response = await http.get(Uri.parse(
          'https://maps.googleapis.com/maps/api/place/nearbysearch/json?keyword=dermatologist&location=$latitude,$longitude&radius=30000&type=hospital&key=AIzaSyC8CnUOQ_fvVfiFTOfmDkEhXnJXzQXQSO0'));
      if (response.statusCode == 200) {
        var result = jsonDecode(response.body);
        log(result.toString());
        mapModel.addAll(RxList<Map<String, dynamic>>.from(result['results'])
            .map((e) => MapModel.fromJson(e))
            .toList());
      } else {
        print('Error fetching data. Status code: ${response.statusCode}');
        print('Response body: ${response.body}');
      }
    } catch (e) {
      print('Error while getting data: $e');
    } finally {
      isLoading(false);
      print('Finally: $mapModel');
      createMarkers();
    }
  }

  createMarkers() {
    mapModel.forEach((element) {
      // Access 'user_ratings_total' field from the API response data

      markers.add(Marker(
        markerId: MarkerId(element.placeId.toString()),
        icon: BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueBlue),
        position: LatLng(element.latitude, element.longitude),
        infoWindow: InfoWindow(title: element.name, snippet: element.vicinity),
        onTap: () {
          print('Marker tapped');
        },
      ));
    });
  }
}
