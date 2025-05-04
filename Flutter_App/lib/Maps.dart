import 'dart:async';

import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:get/get.dart';

import 'package:google_maps_flutter/google_maps_flutter.dart';

import 'Controllers/Map_controller.dart';

class Myapps extends StatefulWidget {
  const Myapps({Key? key}) : super(key: key);

  @override
  State<Myapps> createState() => _MyappState();
}

class _MyappState extends State<Myapps> {
  MapController mapController = Get.put(MapController());
  String? _currentAddress;
  Position? _currentPosition;
  double latitude = 0;
  double longitude = 0;
  FutureOr<void> _openLocationSettings() async {
    await Geolocator.openLocationSettings();
  }

  FutureOr<bool> _handleLocationPermission() async {
    bool serviceEnabled;
    LocationPermission permission;

    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Location services are disabled'),
            content: Text('Please enable location services to use this app.'),
            actions: <Widget>[
              TextButton(
                child: Text('Cancel'),
                onPressed: () {
                  Navigator.of(context).pop();
                },
              ),
              TextButton(
                child: Text('Open Settings'),
                onPressed: () {
                  Navigator.of(context).pop();
                  _openLocationSettings(); // Open location settings
                },
              ),
            ],
          );
        },
      );
    }
    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Location permissions are denied')));
        return false;
      }
    }
    if (permission == LocationPermission.deniedForever) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text(
              'Location permissions are permanently denied, we cannot request permissions.')));
      return false;
    }
    return true;
  }

  FutureOr<void> _getCurrentPosition() async {
    final hasPermission = await _handleLocationPermission();

    if (!hasPermission) return;
    await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high)
        .then((Position position) {
      setState(() => _currentPosition = position);
      print(_currentPosition!.longitude);
      mapController.fetchLocations(
          _currentPosition!.latitude, _currentPosition!.longitude);
    }).catchError((e) {
      debugPrint(e);
    });
  }

  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    _getCurrentPosition();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
          child: Obx(
        () => mapController.mapModel.isNotEmpty
            ? GoogleMap(
                initialCameraPosition: CameraPosition(
                  target: LatLng(mapController.mapModel.first.latitude,
                      mapController.mapModel.first.longitude),
                  zoom: 13,
                ),
                markers: mapController.markers,
              )
            : Center(
                child: CircularProgressIndicator(),
              ),
      )),
    );
  }
}
