// import 'package:flutter/material.dart';
// import 'package:cloud_firestore/cloud_firestore.dart';
// import 'package:firebase_auth/firebase_auth.dart';
// import 'package:firebase_storage/firebase_storage.dart';
// import 'package:flutter/material.dart';
// import 'package:google_fonts/google_fonts.dart';
//
// import 'dart:async';
// import 'dart:io';
//
// import 'package:camera/camera.dart';
// import 'package:percent_indicator/circular_percent_indicator.dart';
//
// import 'ShowImage.dart';
// class TakePictureScreen2 extends StatefulWidget {
//   List<CameraDescription>? cameras;
//   Map<String, dynamic>? data;
//   TakePictureScreen2({super.key, required this.cameras,required this.data});
//
//   @override
//   TakePictureScreenState createState() => TakePictureScreenState();
// }
//
// class TakePictureScreenState extends State<TakePictureScreen2>
//     with TickerProviderStateMixin {
//   late CameraController _controller;
//   late double minzoomoffset = 1;
//   int indicator = 0;
//   late double maxzoomoffset = 10;
//   late FutureOr<void> _initializeControllerFuture;
//   late TabController tabController;
//   TextEditingController email = new TextEditingController();
//   TextEditingController pass = new TextEditingController();
//   TextEditingController conpass = new TextEditingController();
//   int currentTabIndex = 0;
//   double _currentZoomOffset = 1;
//   @override
//   void initState() {
//     super.initState();
//     _controller = CameraController(widget.cameras![0], ResolutionPreset.high,
//         enableAudio: false);
//     // minzoomoffset=_controller.getMaxZoomLevel();
//     // minzoomoffset=_controller.getMinZoomLevel();
//     _controller.initialize().then((_) {
//       // Set autofocus mode
//       _controller.setFlashMode(FlashMode.off);
//       if (!mounted) {
//         return;
//       }
//
//       setState(() {});
//     });
//   }
//
//   @override
//   void dispose() {
//     // Dispose of the controller when the widget is disposed.
//     _controller.dispose();
//     super.dispose();
//   }
//
//   void onTabChange() {
//     setState(() {
//       currentTabIndex = tabController.index;
//     });
//   }
//
//   Map<String, dynamic> responsee = {};
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       backgroundColor: Colors.white,
//       body: SingleChildScrollView(
//         child: Column(
//           children: [
//             SizedBox(
//               height: 30,
//             ),
//             Padding(
//               padding: const EdgeInsets.all(10),
//               child: Container(
//                 decoration:
//                 BoxDecoration(borderRadius: BorderRadius.circular(20)),
//                 child: (_controller.value.isInitialized)
//                     ? Container(
//                     child: Center(
//                       child: Container(
//                         child: ClipRRect(
//                             borderRadius: BorderRadius.circular(16.0),
//                             child: CameraPreview(_controller)),
//                       ),
//                     ))
//                     : CircularProgressIndicator(),
//               ),
//             ),
//             Padding(
//               padding: const EdgeInsets.symmetric(horizontal: 20),
//               child: Row(
//                 mainAxisAlignment: MainAxisAlignment.spaceBetween,
//                 children: [
//                   Text('-', style: TextStyle(fontSize: 40)),
//                   Expanded(
//                     child: Slider(
//                       value: _currentZoomOffset,
//                       min: minzoomoffset,
//                       max: maxzoomoffset,
//                       onChanged: (newv) {
//                         setState(() {
//                           _currentZoomOffset = newv;
//                           print(_currentZoomOffset);
//                           if (newv > minzoomoffset && newv < maxzoomoffset)
//                             _controller.setZoomLevel(newv);
//                         });
//                       },
//                     ),
//                   ),
//                   Text(
//                     '+',
//                     style: TextStyle(fontSize: 30),
//                   ),
//                 ],
//               ),
//             )
//           ],
//         ),
//       ),
//       floatingActionButton: Padding(
//         padding: const EdgeInsets.only(bottom: 30),
//         child: FloatingActionButton.extended(
//           label: Text("Click!"),
//
//           // Provide an onPressed callback.
//           onPressed: () async {
//             //!email.text.isEmpty && !pass.text.isEmpty
//             if (email.text.isEmpty && pass.text.isEmpty && indicator == 0) {
//               try {
//                 // Attempt to take a picture and get the file image
//                 // where it was saved.
//                 setState(() {
//                   indicator = 1;
//                 });
//                 final image = await _controller.takePicture();
//                 Navigator.of(context).push(MaterialPageRoute(
//                     builder: (context) => ShowImage(imagePath: image.path,userType: widget.user,skin : widget.skin,questions: widget.questions,)));
//                 setState(() {
//                   indicator = 0;
//                 });
//                 // showDialog<String>(
//                 //   context: context,
//                 //   builder: (BuildContext context) => AlertDialog(
//                 //     title: Text('there is ${dataMap?["result"][0]}'),
//                 //
//                 //     actions: <Widget>[
//                 //       TextButton(
//                 //         onPressed: () => Navigator.pop(context, 'Cancel'),
//                 //         child: const Text('Cancel'),
//                 //       ),
//                 //       TextButton(
//                 //         onPressed: () => Navigator.pop(context, 'OK'),
//                 //         child: const Text('OK'),
//                 //       ),
//                 //     ],
//                 //   ),
//                 // );
//
//                 if (!mounted) return;
//               } catch (e) {
//                 // If an error occurs, log the error to the console.
//                 print(e);
//                 ScaffoldMessenger.of(context).showSnackBar(SnackBar(
//                   content: const Text('Issue in connecting cameras'),
//                   action: SnackBarAction(
//                     label: 'Undo',
//                     onPressed: () {
//                       // Some code to undo the change.
//                     },
//                   ),
//                 ));
//               }
//             }
//           },
//           icon: Row(
//             mainAxisAlignment: MainAxisAlignment.center,
//             children: [
//               const Icon(Icons.camera_alt),
//               (indicator == 1)
//                   ? Container(
//                   height: 10, width: 10, child: CircularProgressIndicator())
//                   : SizedBox()
//             ],
//           ),
//         ),
//       ),
//       floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
//     );
//   }
//
//   FutureOr<void> _showMyDialog(String targ, String val) async {
//     return showDialog<void>(
//       context: context,
//       barrierDismissible: false, // user must tap button!
//       builder: (BuildContext context) {
//         return AlertDialog(
//           title: const Text('AlertDialog Title'),
//           content: const SingleChildScrollView(
//             child: ListBody(
//               children: <Widget>[],
//             ),
//           ),
//           actions: <Widget>[
//             TextButton(
//               child: const Text('Ok'),
//               onPressed: () {
//                 Navigator.of(context).pop();
//               },
//             ),
//           ],
//         );
//       },
//     );
//   }
// }