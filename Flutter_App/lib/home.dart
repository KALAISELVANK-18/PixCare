import 'package:intl/intl.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:firebase_auth/firebase_auth.dart';

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  // Future<void>? _initializeVideoPlayerFuture;

  String formatTimestamp(int timestamp) {
    var format = new DateFormat('dd-MMM-yyyy');
    var date = new DateTime.fromMillisecondsSinceEpoch(timestamp * 1000);
    return format.format(date);
  }

  @override
  void initState() {
    super.initState();
    super.initState();

    @override
    void dispose() {
      super.dispose();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        elevation: 0,

        // TRY THIS: Try changing the color here to a specific color (to
        // Colors.amber, perhaps?) and trigger a hot reload to see the AppBar
        // change color while the other colors stay the same.
        backgroundColor: Colors.white,
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Text(
              "N",
              style: GoogleFonts.aBeeZee(
                  textStyle:
                      TextStyle(fontWeight: FontWeight.bold, fontSize: 30)),
            ),
            Image.asset(
              "images/nova.png",
              height: 30,
              width: 30,
            ),
            Text("va",
                style: GoogleFonts.kanit(
                  textStyle: TextStyle(fontSize: 25),
                )),
            Padding(
              padding: const EdgeInsets.only(top: 8.0),
              child: Text(
                "l",
                style: GoogleFonts.sacramento(
                  textStyle: TextStyle(
                      fontSize: 27,
                      fontWeight: FontWeight.bold,
                      color: Colors.amber),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 8.0),
              child: Text(
                "a",
                style: GoogleFonts.sacramento(
                  textStyle: TextStyle(
                      fontSize: 27,
                      fontWeight: FontWeight.bold,
                      color: Colors.redAccent),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 8.0),
              child: Text(
                "bs",
                style: GoogleFonts.sacramento(
                  textStyle: TextStyle(
                      fontSize: 27,
                      fontWeight: FontWeight.bold,
                      color: Colors.blue),
                ),
              ),
            ),
          ],
        ),
      ),
      backgroundColor: Colors.white,
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 0),
            child: StreamBuilder<QuerySnapshot>(
                stream: FirebaseFirestore.instance
                    .collection(
                        FirebaseAuth.instance.currentUser!.email.toString())
                    .doc("posts")
                    .collection("posts")
                    .snapshots(),
                builder: (context, AsyncSnapshot<QuerySnapshot> snapshot) {
                  if (snapshot.hasData && snapshot.data != null) {
                    List<DocumentSnapshot> nn = snapshot.data!.docs;
                    return SingleChildScrollView(
                      child: Padding(
                        padding: EdgeInsets.symmetric(horizontal: 10.0),
                        child: Column(
                          children: [
                            (nn.length == 0)
                                ? Center(
                                    child: Padding(
                                      padding: const EdgeInsets.all(20.0),
                                      child: Text("No post found"),
                                    ),
                                  )
                                : SizedBox(),
                            Column(
                              children: [
                                Padding(
                                  padding: const EdgeInsets.all(8.0),
                                  child: Text("Reports",
                                      style: GoogleFonts.aBeeZee(
                                          textStyle: TextStyle(
                                              fontWeight: FontWeight.bold,
                                              fontSize: 17))),
                                ),
                                ListView.builder(
                                  shrinkWrap: true,
                                  physics: NeverScrollableScrollPhysics(),
                                  itemCount: nn.length,
                                  itemBuilder:
                                      (BuildContext context, int index) {
                                    return Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: Container(
                                        child: GestureDetector(
                                          onTap: () {
                                            //Navigator.of(context).push(MaterialPageRoute(builder: (context)=>MyHomePage()));
                                          },
                                          child: Container(
                                            decoration: BoxDecoration(
                                                color: Color.fromARGB(
                                                    255, 255, 219, 172),
                                                borderRadius:
                                                    BorderRadius.circular(10)),
                                            child: Row(
                                              mainAxisAlignment:
                                                  MainAxisAlignment.start,
                                              children: [
                                                Padding(
                                                  padding:
                                                      const EdgeInsets.all(8.0),
                                                  child: Column(
                                                    children: [
                                                      Padding(
                                                        padding:
                                                            const EdgeInsets
                                                                .all(15.0),
                                                        child: Container(
                                                            height: 100,
                                                            width: 100,
                                                            decoration:
                                                                BoxDecoration(
                                                              borderRadius:
                                                                  BorderRadius
                                                                      .circular(
                                                                          10),
                                                              image:
                                                                  DecorationImage(
                                                                fit: BoxFit
                                                                    .cover,
                                                                image:
                                                                    NetworkImage(
                                                                  nn[index][
                                                                      "picture_url"],
                                                                  //"https://images.unsplash.com/photo-1566438480900-0609be27a4be?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8aW1hZ2V8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&w=600&q=60"
                                                                ),
                                                              ),
                                                            ),
                                                            child: SizedBox()),
                                                      ),
                                                    ],
                                                  ),
                                                ),
                                                Padding(
                                                  padding:
                                                      const EdgeInsets.only(
                                                          right: 0),
                                                  child: Column(
                                                    mainAxisAlignment:
                                                        MainAxisAlignment
                                                            .spaceBetween,
                                                    crossAxisAlignment:
                                                        CrossAxisAlignment
                                                            .start,
                                                    children: [
                                                      Container(
                                                          width: 180,
                                                          child: Text(
                                                              nn[index]
                                                                  ["title"],
                                                              textAlign:
                                                                  TextAlign
                                                                      .start,
                                                              style: GoogleFonts
                                                                  .aBeeZee(
                                                                      textStyle:
                                                                          TextStyle(
                                                                fontWeight:
                                                                    FontWeight
                                                                        .bold,
                                                              )))),
                                                      Row(
                                                        children: [
                                                          Icon(Icons
                                                              .calendar_month),
                                                          Container(
                                                              width: 180,
                                                              child: Text(
                                                                  formatTimestamp(nn[
                                                                              index]
                                                                          [
                                                                          "time"]
                                                                      .seconds),
                                                                  style: GoogleFonts
                                                                      .aBeeZee(
                                                                          textStyle:
                                                                              TextStyle(
                                                                    fontWeight:
                                                                        FontWeight
                                                                            .bold,
                                                                  )))),
                                                        ],
                                                      )
                                                    ],
                                                  ),
                                                )
                                              ],
                                            ),
                                          ),
                                        ),
                                      ),
                                    );
                                  },
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    );
                  } else {
                    return CircularProgressIndicator();
                  }
                }),
          ),
        ],
      ),
    );
  }
}
