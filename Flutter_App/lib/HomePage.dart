// import 'package:camera/camera.dart';
// import 'package:cloud_firestore/cloud_firestore.dart';
// import 'package:firebase_auth/firebase_auth.dart';
// import 'package:flutter/cupertino.dart';
// import 'package:flutter/material.dart';
// import 'package:carousel_slider/carousel_slider.dart';
// import 'package:intl/intl.dart';
// import 'package:novalabs/Report.dart';
// import 'package:novalabs/camera.dart';
// import 'package:pie_chart/pie_chart.dart';

// List<CameraDescription>? cameras;

// class HomePage extends StatefulWidget {
//   const HomePage({Key? key}) : super(key: key);

//   @override
//   State<HomePage> createState() => _HomePageState();
// }

// class _HomePageState extends State<HomePage> {
//   int _currentIndex = 0;
  
//   final List<String> sliderItems = [
//     "list.png",
//     "report.png",
//     "3dmap.jpg",
//   ];

//   final dataMap = <String, double>{
//     "Disease Found": 10,
//     "No Disease Found": 10,
//   };

//   final colorList = <Color>[Colors.greenAccent, Colors.redAccent];
//   String formatTimestamp(int timestamp) {
//     var format = new DateFormat('dd-MMM-yyyy');
//     var date = new DateTime.fromMillisecondsSinceEpoch(timestamp * 1000);
//     return format.format(date);
//   }

//   @override
//   void initState() {
//     super.initState();
//     super.initState();

//     @override
//     void dispose() {
//       super.dispose();
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       // appBar: AppBar(
//       //   title: Text("Your Personal AI Dermatologist"),
//       // ),
//       body: Container(
//         decoration: BoxDecoration(
//           image: DecorationImage(
//             image: AssetImage("images/bg.png"),
//             fit: BoxFit.cover,
//           ),
//         ),
//         child: SingleChildScrollView(
//           // Wrap the entire content in SingleChildScrollView
//           child: Column(
//             children: [
//               Padding(
//                 padding: const EdgeInsets.only(top: 40),
//                 child: Row(
//                   children: [
//                     Padding(
//                       padding: const EdgeInsets.symmetric(
//                           horizontal: 20, vertical: 10),
//                       child: Text(
//                         "AI Dermatologist",
//                         softWrap: true,
//                         style: TextStyle(
//                             color: Colors.white,
//                             fontWeight: FontWeight.w600,
//                             fontSize: 30),
//                       ),
//                     ),
//                     // Column(
//                     //   children: [
//                     //     Text(
//                     //       "Your Personal",
//                     //       style: TextStyle(
//                     //           color: Colors.white,
//                     //           fontWeight: FontWeight.w400,
//                     //           fontSize: 30),
//                     //     ),
//                     //     Padding(
//                     //       padding: const EdgeInsets.symmetric(horizontal: 20),
//                     //       child: Text(
//                     //         "AI Dermatologist",
//                     //         style: TextStyle(
//                     //             color: Colors.white,
//                     //             fontWeight: FontWeight.w600,
//                     //             fontSize: 30),
//                     //       ),
//                     //     ),
//                     //   ],
//                     // )
//                   ],
//                 ),
//               ),
//               Padding(
//                 padding: const EdgeInsets.only(top: 20),
//                 child: Positioned(
//                   top: 40,
//                   left: 0,
//                   right: 0,
//                   child: Column(
//                     mainAxisAlignment: MainAxisAlignment.start,
//                     crossAxisAlignment: CrossAxisAlignment.center,
//                     children: [
//                       CarouselSlider(
//                         options: CarouselOptions(
//                           height: 200.0,
//                           enlargeCenterPage: true,
//                           autoPlay: true,
//                           aspectRatio: 16 / 9,
//                           autoPlayInterval: Duration(seconds: 3),
//                           autoPlayAnimationDuration:
//                               Duration(milliseconds: 800),
//                           enableInfiniteScroll: true,
//                           autoPlayCurve: Curves.fastOutSlowIn,
//                           viewportFraction: 0.8,
//                         ),
//                         items: sliderItems.map((item) {
//                           return Builder(
//                             builder: (BuildContext context) {
//                               return Container(
//                                 width: MediaQuery.of(context).size.width,
//                                 margin: EdgeInsets.symmetric(horizontal: 5.0),
//                                 decoration: BoxDecoration(
//                                     image: DecorationImage(
//                                         image: AssetImage("images/$item"),
//                                         fit: BoxFit.cover),
//                                     color: Colors.white,
//                                     borderRadius: BorderRadius.circular(10)),
//                                 child: Center(),
//                               );
//                             },
//                           );
//                         }).toList(),
//                       ),
//                     ],
//                   ),
//                 ),
//               ),
//               Positioned(
//                 child: Padding(
//                   padding: const EdgeInsets.only(top: 20),
//                   child: Container(
//                     height: 1100,
//                     width: MediaQuery.of(context).size.width * 1,
//                     decoration: BoxDecoration(
//                       color: Color.fromRGBO(229, 235, 255, 1),
//                       borderRadius: BorderRadius.only(
//                         topLeft: Radius.circular(30),
//                         topRight: Radius.circular(30),
//                       ),
//                     ),
//                     child: Padding(
//                       padding: const EdgeInsets.all(15),
//                       child: Column(
//                         crossAxisAlignment: CrossAxisAlignment.start,
//                         children: [
//                           Text(
//                             "Early Detection ",
//                             style: TextStyle(
//                               fontSize: 22,
//                               fontWeight: FontWeight.w600,
//                             ),
//                           ),
//                           Text(
//                             "Makes a Difference",
//                             style: TextStyle(
//                                 fontSize: 22, fontWeight: FontWeight.w600),
//                           ),
//                           Padding(
//                             padding: const EdgeInsets.only(top: 10),
//                             child: Container(
//                               height: 350,
//                               width: MediaQuery.of(context).size.width * 1.0,
//                               decoration: BoxDecoration(
//                                 color: Colors.white,
//                                 borderRadius: BorderRadius.circular(10),
//                               ),
//                               child: Column(
//                                 children: [
//                                   Padding(
//                                     padding: const EdgeInsets.all(12),
//                                     child: Container(
//                                       height: 100,
//                                       width: MediaQuery.of(context).size.width *
//                                           1.0,
//                                       decoration: BoxDecoration(
//                                         color: Color.fromRGBO(86, 128, 227, 1),
//                                         borderRadius: BorderRadius.circular(8),
//                                       ),
//                                       child: Padding(
//                                         padding: const EdgeInsets.all(22),
//                                         child: Row(
//                                           children: [
//                                             Column(
//                                               children: [
//                                                 Text(
//                                                   "Our test can help you",
//                                                   style: TextStyle(
//                                                       color: Colors.white,
//                                                       fontSize: 19,
//                                                       fontWeight:
//                                                           FontWeight.w500),
//                                                 ),
//                                                 Text(
//                                                   "to detect Melanoma.",
//                                                   style: TextStyle(
//                                                       color: Colors.white,
//                                                       fontSize: 19,
//                                                       fontWeight:
//                                                           FontWeight.w500),
//                                                 ),
//                                               ],
//                                             ),
//                                             Padding(
//                                               padding: const EdgeInsets.only(
//                                                   left: 60),
//                                               child: Image.asset(
//                                                 "images/scan2.png",
//                                               ),
//                                             ),
//                                           ],
//                                         ),
//                                       ),
//                                     ),
//                                   ),
//                                   Padding(
//                                     padding: const EdgeInsets.all(10),
//                                     child: Text(
//                                       "Finding melanoma at an early stage is crucial; Early detection can vastly increase your chances for cure.",
//                                       softWrap: true,
//                                       style: TextStyle(
//                                         fontSize: 16,
//                                         color: Color.fromRGBO(128, 128, 128, 1),
//                                       ),
//                                     ),
//                                   ),
//                                   Padding(
//                                     padding: const EdgeInsets.all(10),
//                                     child: Text(
//                                       "Most moles, brown spots and growths on the skin are harmless - but not always.",
//                                       softWrap: true,
//                                       style: TextStyle(
//                                         fontSize: 16,
//                                         color: Color.fromRGBO(147, 147, 147, 1),
//                                       ),
//                                     ),
//                                   ),
//                                   OutlinedButton(
//                                     onPressed: () {
//                                       // Add your button logic here
//                                     },
//                                     child: Text(
//                                       "Read More",
//                                       style: TextStyle(
//                                         fontSize: 14,
//                                       ),
//                                     ),
//                                   ),
//                                 ],
//                               ),
//                             ),
//                           ),
//                           Padding(
//                             padding: const EdgeInsets.only(top: 10, left: 10),
//                             child: Text(
//                               "Your last scanning",
//                               style: TextStyle(
//                                 fontSize: 22,
//                                 fontWeight: FontWeight.w600,
//                               ),
//                             ),
//                           ),
//                           Container(
//                             child: StreamBuilder<QuerySnapshot>(
//                               stream: FirebaseFirestore.instance
//                                   .collection(FirebaseAuth
//                                       .instance.currentUser!.email
//                                       .toString())
//                                   .doc("posts")
//                                   .collection("posts")
//                                   .snapshots(),
//                               builder: (context,
//                                   AsyncSnapshot<QuerySnapshot> snapshot) {
//                                 if (snapshot.hasData && snapshot.data != null) {
//                                   List<DocumentSnapshot> nn =
//                                       snapshot.data!.docs;
//                                   // Calculate the index to get the last two items
//                                   int startIndex =
//                                       nn.length >= 2 ? nn.length - 2 : 0;

//                                   return GridView.builder(
//                                     shrinkWrap: true,
//                                     physics: NeverScrollableScrollPhysics(),
//                                     gridDelegate:
//                                         SliverGridDelegateWithFixedCrossAxisCount(
//                                       crossAxisCount: 2,
//                                       crossAxisSpacing: 10,
//                                       mainAxisSpacing: 10,
//                                     ),
//                                     itemCount: nn.length >= 2
//                                         ? 2
//                                         : nn.length, // Show the last two items or all if less than two
//                                     itemBuilder:
//                                         (BuildContext context, int index) {
//                                       // Calculate the index based on the last two items
//                                       int itemIndex = startIndex + index;

//                                       return GestureDetector(
//                                         onTap: () {
//                                           // Navigate to a detailed view
//                                         },
//                                         child: Container(
//                                           decoration: BoxDecoration(
//                                             borderRadius:
//                                                 BorderRadius.circular(10),
//                                           ),
//                                           child: Column(
//                                             children: [
//                                               Padding(
//                                                 padding:
//                                                     const EdgeInsets.all(0),
//                                                 child: Container(
//                                                   height: 140,
//                                                   width: 140,
//                                                   decoration: BoxDecoration(
//                                                     borderRadius:
//                                                         BorderRadius.circular(
//                                                             10),
//                                                     image: DecorationImage(
//                                                       fit: BoxFit.cover,
//                                                       image: NetworkImage(
//                                                         nn[itemIndex]
//                                                             ["picture_url"],
//                                                       ),
//                                                     ),
//                                                   ),
//                                                 ),
//                                               ),
//                                               Padding(
//                                                 padding: const EdgeInsets.only(
//                                                   right: 0,
//                                                 ),
//                                                 child: Padding(
//                                                   padding:
//                                                       const EdgeInsets.only(
//                                                           top: 10, left: 25),
//                                                   child: Column(
//                                                     mainAxisAlignment:
//                                                         MainAxisAlignment
//                                                             .spaceBetween,
//                                                     crossAxisAlignment:
//                                                         CrossAxisAlignment
//                                                             .start,
//                                                     children: [
//                                                       // Text(
//                                                       //   nn[itemIndex]["title"],
//                                                       //   textAlign:
//                                                       //       TextAlign.start,
//                                                       //   style: TextStyle(
//                                                       //     fontWeight:
//                                                       //         FontWeight.bold,
//                                                       //   ),
//                                                       // ),
//                                                       Row(
//                                                         children: [
//                                                           Icon(Icons
//                                                               .calendar_month),
//                                                           Text(
//                                                             formatTimestamp(
//                                                                 nn[itemIndex]
//                                                                         ["time"]
//                                                                     .seconds),
//                                                             style: TextStyle(
//                                                               fontWeight:
//                                                                   FontWeight
//                                                                       .bold,
//                                                             ),
//                                                           ),
//                                                         ],
//                                                       )
//                                                     ],
//                                                   ),
//                                                 ),
//                                               )
//                                             ],
//                                           ),
//                                         ),
//                                       );
//                                     },
//                                   );
//                                 } else {
//                                   // Handle loading or error state
//                                   return CircularProgressIndicator(); // You can replace this with an appropriate loading widget
//                                 }
//                               },
//                             ),
//                           ),

//                           // GridView.builder(
//                           //   shrinkWrap: true,
//                           //   physics:
//                           //       NeverScrollableScrollPhysics(), // Disable GridView's scroll
//                           //   gridDelegate:
//                           //       SliverGridDelegateWithFixedCrossAxisCount(
//                           //     crossAxisCount: 2,

//                           //     crossAxisSpacing:
//                           //         10, // Add horizontal spacing between grid items
//                           //     mainAxisSpacing:
//                           //         10, // Number of columns in the grid
//                           //   ),
//                           //   itemCount: 2, // Number of grid items
//                           //   itemBuilder: (BuildContext context, int index) {
//                           //     // Create a grid item here
//                           //     return Card(
//                           //       color: Colors.white,
//                           //       margin: EdgeInsets.all(8),
//                           //       child: ListTile(
//                           //         title: Text('Item $index'),
//                           //         onTap: () {
//                           //           // Add item tap logic here
//                           //         },
//                           //       ),
//                           //     );
//                           //   },
//                           // ),

//                           //
//                           Padding(
//                             padding: const EdgeInsets.all(8.0),
//                             child: Container(
//                               height: 180,
//                               width: MediaQuery.of(context).size.width * 1,
//                               decoration: BoxDecoration(
//                                   color: Colors.white,
//                                   borderRadius: BorderRadius.circular(10)),
//                               child: Padding(
//                                 padding: const EdgeInsets.all(20),
//                                 child: PieChart(
//                                   dataMap: dataMap,
//                                   chartType: ChartType.ring,
//                                   baseChartColor: Colors.blueAccent,
//                                   colorList: colorList,
//                                   chartValuesOptions: ChartValuesOptions(
//                                     showChartValuesInPercentage: true,
//                                   ),
//                                   totalValue: 20,
//                                 ),
//                               ),
//                             ),
//                           ),
//                         ],
//                       ),
//                     ),
//                   ),
//                 ),
//               )
//             ],
//           ),
//         ),
//       ),
//       bottomNavigationBar: BottomNavigationBar(
//         currentIndex: _currentIndex,
//         type: BottomNavigationBarType.fixed,
//         selectedItemColor: Colors.blue,
//         unselectedItemColor: Colors.grey,
//         items: [
//           BottomNavigationBarItem(
//             icon: Icon(Icons.home),
//             label: 'Home',
//           ),
//           BottomNavigationBarItem(
//             icon: Icon(Icons.search),
//             label: 'Search',
//           ),
//           BottomNavigationBarItem(
//             icon: SizedBox(
//               width: 60.0,
//               height: 60.0,
//               child: FloatingActionButton(
//                 onPressed: () {
//                   Navigator.push(
//                       context,
//                       MaterialPageRoute(
//                           builder: (context) => TakePictureScreen(
//                                 cameras: cameras,
//                               )));
//                 },
//                 child: Icon(
//                   Icons.camera,
//                   size: 36.0,
//                 ),
//                 backgroundColor: Colors.blue,
//               ),
//             ),
//             label: 'Camera',
//           ),
//           BottomNavigationBarItem(
//             icon: IconButton(
//               icon: Icon(Icons.notifications),
//               onPressed: () {
//                 Navigator.of(context).push(
//                   MaterialPageRoute(
//                     builder: (context) => Report(),
//                   ),
//                 );
//               },
//             ),
//             label: 'Reports',
//           ),
//           BottomNavigationBarItem(
//             icon: Icon(Icons.person),
//             label: 'Profile',
//           ),
//         ],
//         onTap: (index) {
//           setState(() {
//             _currentIndex = index;
//           });
//         },
//       ),
//     );
//   }
// }
